'''
@author: enpei
@date:2022-09-29
简单的PID控制类，用来实现无人机高度、旋转角度等

'''
import time


class simple_PID:

    '''
    简单的PID类
    '''

    def __init__(self, pid_paras_list):
        # 初始化配置参数
        self.kp = pid_paras_list[0]
        self.ki = pid_paras_list[1]
        self.kd = pid_paras_list[2]
        # 上一次记录的误差
        self.previous_error = 0
        # 上一次记录时间，用来计算dt
        self.previous_record_time = time.time()
        # 积分
        self.integral = 0

    def setParas(self, which='p', add_val=0.01):
        
        '''
        重新设置参数（用来微调测试）
        '''

        if which == 'p':
            self.kp = round(self.kp + add_val, 2)
        elif which == 'i':
            self.ki = round(self.ki + add_val, 2)
        elif which == 'd':
            self.kd = round(self.kd + add_val, 2)

        # 重新初始化参数
        self.previous_error = 0
        self.previous_record_time = time.time()
        self.integral = 0

        # 返回变更后的参数列表
        return [self.kp, self.ki, self.kd]

    def update(self, current_error,min_val=-100,max_val=100):
        '''
        更新参数，得到输出值
        '''
        # 1.误差：观测值，传过来的参数
        error = current_error
        # 2.微分：速度 = 距离/dt
        # dt：随着电脑处理速度可能会有扰动，需要计算出来
        now = time.time()
        dt = (now-self.previous_record_time)
        derivative = (error - self.previous_error) / dt
        # 3.积分
        self.integral = self.integral + error * dt

        # 更新状态
        self.previous_error = error
        self.previous_record_time = now

        # 计算结果
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        # 控制输出的最大值最小值
        if output > max_val:
            output = max_val
        if output < min_val:
            output = min_val

        return output
