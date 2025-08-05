
class Counter:
    def __init__(self,
                 all_steps):
        "初始化一个需要迭代all_steps的计数器"
        self.steps_all = all_steps
        self.current_step = 0
        self.finished = False

    @property
    def percent(self):
        return self.current_step/self.steps_all

    def step(self):
        """每次调用增加进度,并根据进度产生相应位置的结果
        返回1,2,...,end,不包括start
        """
        if self.current_step < self.steps_all:
            self.current_step = self.current_step+1
            if self.current_step == self.steps_all:
                self.finished = True
        else:
            #已经完成了但还是调用
            raise Exception

        return self.current_step