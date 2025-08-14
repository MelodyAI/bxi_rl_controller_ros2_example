#include <iostream>
#include <communication/msg/motion_commands.hpp>
#include <linux/joystick.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include "rclcpp/rclcpp.hpp"

using namespace std::chrono_literals;
using namespace std;

#if 0  //PS4 JS
#define JS_VELX_AXIS 4
#define JS_VELX_AXIS_DIR -1
#define JS_VELY_AXIS 0
#define JS_VELY_AXIS_DIR -1
#define JS_VELR_AXIS 6
#define JS_VELR_AXIS_DIR -1

#define JS_STOP_BT 10
#define JS_GAIT_STAND_BT 0
#define JS_GAIT_WALK_BT 2
#define JS_HEIGHT_UPPER_BT  1
#define JS_HEIGHT_LOWER_BT  3
#define JS_MODE_BT          5
#define JS_START_BT 9
#else //XBOX JS
#define JS_VELX_AXIS 3
#define JS_VELX_AXIS_DIR -1
#define JS_VELY_AXIS 0
#define JS_VELY_AXIS_DIR -1
#define JS_VELR_AXIS 6
#define JS_VELR_AXIS_DIR -1
#define JS_STOP_BT 11
#define JS_START_BT 13
#define JS_SHOULDER_LEFT_BT  6
#define JS_SHOULDER_RIGHT_BT  7
#define JS_Y_BT 4
#define JS_X_BT 3
#define JS_A_BT 0
#define JS_B_BT 1
#endif

#define AXIS_DEAD_ZONE  1000

#define MIN_SPEED_X -0.8
#define MAX_SPEED_X 0.8
#define MIN_SPEED_Y -0.4
#define MAX_SPEED_Y 0.4
#define MIN_SPEED_R -0.6
#define MAX_SPEED_R 0.6

#define AXIS_VALUE_MAX 32767

#define STAND_HEIGHT 1.0
#define STAND_HEIGHT_MIN    1.0
#define STAND_HEIGHT_MAX    3.0

class COMPublisher : public rclcpp::Node{
public:
    COMPublisher(const char *_js_dev) : Node("COM_publisher"){
        if (strlen(_js_dev) >= 128){
            printf("dev:%s error\n", _js_dev);
            exit(-1);
        }

        strcpy(_js_dev_name, _js_dev);
        
        while (1){
            js_fd = open(_js_dev_name, O_RDONLY); // O_NONBLOCK
            if (js_fd < 0){
                printf("open:%s failed\n", _js_dev_name);
                sleep(1);      
            }
            else{
                printf("open js dev: %s\n", _js_dev_name);
                break;
            }
        }
        
        com_pub = this->create_publisher<communication::msg::MotionCommands>("motion_commands", 20);
        timer_ = this->create_wall_timer(10ms, std::bind(&COMPublisher::timer_callback, this));
        js_loop_thread_ = std::thread(&COMPublisher::js_loop, this);
    }

    ~COMPublisher(){
        if (js_fd > 0){
            close(js_fd);
        }
    }

private:
    mutable std::mutex lock_;

    char _js_dev_name[128] = {0};
    int js_fd;
    double js_axis[20] = {0};   //原始数据
    double js_bt[20] = {0};
    std::thread js_loop_thread_;

    double velxy[2] = {0};      //x y速度
    double velxy_filt[2] = {0}; //x y速度滤波值
    double stand_height = STAND_HEIGHT;
    double height_filt = STAND_HEIGHT;
    double velr = 0;    //旋转速度
    double velr_filt = 0;
    double vel_offset = 0.0;

    bool high_jump_toggle = false;  // 跳高/发送障碍高程图(Y)
    bool far_jump_toggle = false;  // 立定跳远(X)
    bool stop_btn_toggle = false;  // B
    bool dance_toggle = false;  // 舞蹈(A)

    void timer_callback(){
        auto message = communication::msg::MotionCommands();{
            const std::lock_guard<std::mutex> guard(lock_);

            velxy[0] = (js_axis[JS_VELX_AXIS] * JS_VELX_AXIS_DIR) / (double)AXIS_VALUE_MAX;
            velxy[1] = (js_axis[JS_VELY_AXIS] * JS_VELY_AXIS_DIR) / (double)AXIS_VALUE_MAX;
            velr = (js_axis[JS_VELR_AXIS] * JS_VELR_AXIS_DIR) / (double)AXIS_VALUE_MAX;

            velxy[0] = fabs(velxy[0]) > AXIS_DEAD_ZONE / (double)AXIS_VALUE_MAX ? velxy[0] : 0;
            velxy[1] = fabs(velxy[1]) > AXIS_DEAD_ZONE / (double)AXIS_VALUE_MAX ? velxy[1] : 0;
            velr = fabs(velr) > AXIS_DEAD_ZONE / (double)AXIS_VALUE_MAX ? velr : 0;
            
            //按定义最大速度缩放
            if (velxy[0] > 0){
                velxy[0] *= MAX_SPEED_X;
            }
            else if (velxy[0] < 0){
                velxy[0] *= -MIN_SPEED_X;
            }

            if (velxy[1] > 0){
                velxy[1] *= MAX_SPEED_Y;
            }
            else if (velxy[1] < 0){
                velxy[1] *= -MIN_SPEED_Y;
            }

            if (velr > 0){
                velr *= MAX_SPEED_R;
            }
            else if (velr < 0){
                velr *= -MIN_SPEED_R;
            }

            velxy_filt[0] = velxy[0] * 0.03 + velxy_filt[0] * 0.97;
            velxy_filt[1] = velxy[1] * 0.03 + velxy_filt[1] * 0.97;

            velr_filt = velr * 0.05 + velr_filt *  0.95;

            message.vel_des.x = velxy_filt[0] + vel_offset;
            message.vel_des.y = velxy_filt[1];
            message.yawdot_des = velr_filt;

            // 设置手臂控制标志
            message.btn_5 = dance_toggle ? 1 : 0; // A
            message.btn_6 = far_jump_toggle ? 1 : 0; // X
            message.btn_7 = high_jump_toggle ? 1 : 0; // Y
            message.btn_10 = stop_btn_toggle ? 1 : 0; // B

            height_filt = height_filt * 0.9 + stand_height * 0.1;
            message.height_des = height_filt;
        }

        com_pub->publish(message);
    }

    void reset_value()
    {
        const std::lock_guard<std::mutex> guard(lock_);
        memset(js_axis, 0, sizeof(js_axis));
        memset(velxy, 0, sizeof(velxy));
        memset(velxy_filt, 0, sizeof(velxy_filt));
        velr_filt = 0;
        height_filt = STAND_HEIGHT;
    }

    void js_loop(){
        while (1){
            ssize_t len;
            struct js_event event;

            len = read(js_fd, &event, sizeof(event));

            if (len == sizeof(event)){
                if (event.type & JS_EVENT_AXIS){
                    //printf("Axis: %d -> %d\n", (int)event.number, (int)event.value);
                    js_axis[event.number] = event.value;
                }
                else if (event.type & JS_EVENT_BUTTON){
                    //printf("Button: %d -> %d\n", (int)event.number, (int)event.value);
                    if (event.value){
                        switch (event.number){
                        case JS_STOP_BT:{
                            system("killall -SIGINT robot_controller");
                            system("killall -SIGINT pt_main_thread");
                            system("killall -SIGINT bxi_example_py");
                            system("killall -SIGINT bxi_example_py_trunk");
                            system("killall -SIGINT bxi_example_py_ankle");
                            system("killall -SIGINT hardware");
                            system("killall -SIGINT hardware_trunk");
                            system("killall -SIGINT hardware_trunk_neck");
                            system("killall -SIGINT hardware_ankle");
                            system("killall -SIGINT xuxin_controller_terrain");
                            system("killall -SIGINT simulation");// mujoco
                            printf("kill robot_controller\n");//robot_controller

                            reset_value();
                        }
                        break;
                        case JS_START_BT:{
                            system("mkdir -p /var/log/bxi_log");
                            system("ros2 launch bxi_example_py_ankle example_launch_hw.py > /var/log/bxi_log/$(date +%Y-%m-%d_%H-%M-%S)_elf.log  2>&1 &");
                            printf("run robot\n");//robot_controller

                            reset_value();
                        }
                        break;
                        case JS_SHOULDER_LEFT_BT:{
                            const std::lock_guard<std::mutex> guard(lock_);
                            stand_height -= 0.2;
                            if (stand_height > STAND_HEIGHT_MAX)
                            {
                                stand_height = STAND_HEIGHT_MAX;
                            }
                            printf("stand_height: %f\n", stand_height);
                        }
                        break;
                        case JS_SHOULDER_RIGHT_BT:{
                            const std::lock_guard<std::mutex> guard(lock_);
                            stand_height += 0.2;
                            if (stand_height < STAND_HEIGHT_MIN)
                            {
                                stand_height = STAND_HEIGHT_MIN;
                            }
                            printf("stand_height: %f\n", stand_height);
                        }
                        break;
                        case JS_A_BT:{
                            const std::lock_guard<std::mutex> guard(lock_);
                            dance_toggle = !dance_toggle;
                            printf("dance\n");
                        }
                        break;
                        case JS_X_BT:{
                            const std::lock_guard<std::mutex> guard(lock_);
                            far_jump_toggle = !far_jump_toggle;
                            printf("far jump\n");
                        }
                        break;
                        case JS_Y_BT:{
                            const std::lock_guard<std::mutex> guard(lock_);
                            high_jump_toggle = !high_jump_toggle;
                            printf("high jump\n");
                        }
                        break;
                        case JS_B_BT:{
                            const std::lock_guard<std::mutex> guard(lock_);
                            stop_btn_toggle = !stop_btn_toggle;
                            printf("stop\n");
                        }
                        break;
                        default:
                            break;
                        }
                    }
                }
                else{
                    printf("unknown event:%u\n", event.type);
                }
            }
            if (len <= 0){
                printf("js dev lost, retry\n");
                close(js_fd);
                while (1){
                    js_fd = open(_js_dev_name, O_RDONLY); // O_NONBLOCK
                    if (js_fd < 0){
                        printf("open:%s failed\n", _js_dev_name);
                        sleep(1);
                    }
                    else{
                        printf("open js dev: %s\n", _js_dev_name);
                        break;
                    }
                }
            }
        }
    }

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<communication::msg::MotionCommands>::SharedPtr com_pub;
};

int main(int argc, const char *argv[]){
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<COMPublisher>("/dev/input/js0"));
    rclcpp::shutdown();

    return 0;
}
