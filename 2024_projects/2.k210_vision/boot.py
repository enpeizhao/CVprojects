import sensor, image, lcd, time

import KPU as kpu
import gc, sys
from machine import Timer, PWM


# 初始化定时器和PWM对象
tim = Timer(Timer.TIMER0, Timer.CHANNEL0, mode=Timer.MODE_PWM)
beep = PWM(tim, freq=1, duty=50, pin=9)
beep.disable()

def play_prepare():
    beep.enable()
    """进入番茄钟，播放升调音符"""
    beep.freq(1000)
    time.sleep(0.2)
    beep.freq(2000)
    time.sleep(0.2)
    beep.disable()

def play_tomato():
    beep.enable()
    """进入番茄钟，播放升调音符"""
    for freq in [400, 500, 600]: # 播放5个升调音符
        beep.freq(freq)
        time.sleep(0.2) # 每个音符持续0.2秒
    beep.disable()

def play_interrupt():
    beep.enable()
    """中断提醒，播放降调音符"""
    for freq in [1600, 1500]: # 播放两个降调音符
        beep.freq(freq)
        time.sleep(0.2) # 每个音符持续0.2秒
    beep.disable()
def play_success():
    beep.enable()
    """番茄钟成功，播放升调音符"""
    for freq in [1200, 1300, 1400]: # 播放三个升调音符
        beep.freq(freq)
        time.sleep(0.2) # 每个音符持续0.33秒
    beep.disable()
# 异常信息
def lcd_show_except(e):
    import uio
    err_str = uio.StringIO()
    sys.print_exception(e, err_str)
    err_str = err_str.getvalue()
    img = image.Image(size=(224,224))
    img.draw_string(0, 10, err_str, scale=1, color=(0xff,0x00,0x00))
    lcd.display(img)

# 主程序
def main():

    labels = ['person','other']
    anchors =  [2.625, 4.65625, 1.34375, 3.375, 0.28125, 0.6875, 4.78125, 6.1875, 0.65625, 1.6875]
    sensor_window=(224, 224)
    # 初始化LCD，invert
    lcd.init(type=1,invert=1)
    # 摄像头配置
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.QVGA)
    #sensor.set_windowing(sensor_window)
    #设置摄像头垂直翻转
    sensor.set_vflip(0)
    sensor.set_hmirror(0)
    #1 表示开始抓取图像 0 表示停止抓取图像
    sensor.run(1)

    # 加载检测模型
    task = kpu.load("/sd/person1.kmodel")
    # 初始化模型
    kpu.init_yolo2(task, 0.4, 0.3, 5, anchors)
    # 比例
    ratio = 320/224
    # 工作模式：0、1、2分别是准备、番茄、休息
    mode = 0
    # 上一次检测到工作状态的时间
    last_detection_time = time.time()
    # 状态初始时间
    state_initial_time = last_detection_time
    # 番茄开始时间
    fanqie_start_time = 0
    # 今日番茄数
    fanqie_total_count = 0
    # 今日番茄打断数
    fanqie_break_count = 0

    # 番茄时长（单位：秒，建议25min）
    fanqie_target_duration = 25*60
    # 休息时长（单位：秒，建议5min）
    reset_target_duration = 5 * 60



    try:
        while 1:
            # 获取图像
            img = sensor.snapshot()
            # resize to 224
            yolo_input = img.resize(224,224)
            # 复制到KPU
            yolo_input.pix_to_ai()


            # 推理
            t = time.ticks_ms()
            objects = kpu.run_yolo2(task, yolo_input)
            t = time.ticks_ms() - t

            # 如果是休息模式，则倒计时直到准备模式
            # 休息模式无需检测人
            if mode ==2:
                current_time = time.time()
                total_duration = current_time - state_initial_time
                if total_duration > reset_target_duration:
                    # 恢复成准备模式
                    mode = 0
                    # 重置
                    state_initial_time = time.time()
                    play_prepare()


            # 检测到工作状态
            if objects:
                # 当前检测时间
                current_detection_time = time.time()
                # 重置上次检测时间
                last_detection_time = time.time()
                #print(duration,state_duration)

                # 如果是准备模式，等待进入番茄模式
                if mode == 0:
                    # 状态持续时间
                    state_duration = current_detection_time - state_initial_time
                    # 间隔小于1S且工作场景检测超过5s，进入番茄模式
                    if state_duration > 5:
                        mode =1
                        play_tomato()
                        # 重置state_initial_time，为番茄模式准备
                        state_initial_time = time.time()
                        # 番茄开始时间，用于计算总共的番茄时间
                        fanqie_start_time = state_initial_time

                # 如果是番茄模式，需要判断：准备模式、番茄、休息模式
                elif mode == 1:
                    # 状态持续时间
                    state_duration = current_detection_time - state_initial_time
                    # 间隔小于1S且工作场景检测超过25min，进入休息模式
                    if state_duration > fanqie_target_duration:
                        mode =2
                        # 重置state_initial_time
                        state_initial_time = time.time()
                        # 增加今日番茄数
                        fanqie_total_count +=1
                        play_success()

                # 遍历并绘制
                for obj in objects:
                    x,y,w,h = obj.rect()
                    xx = int(ratio * x)
                    ww = int(ratio * w)
                    # 检测框
                    img.draw_rectangle(xx, y, ww, h, color = (0, 255, 0), thickness = 4, fill = False)
                    # 标签
                    img.draw_string(xx, y-30, "%s : %.2f" %(labels[obj.classid()], obj.value()), scale=2, color=(0, 255, 0))

            # 注意层级
            else:
                if mode == 0:
                    # 检测间隔大于1s，保持准备模式，重置state_initial_time
                    if (time.time() - last_detection_time )> 1:
                        # 5S计时作废，重置状态计时
                        state_initial_time = time.time()
                # 番茄模式下，检测不到人，超过10秒就中断
                if mode == 1:
                    if (time.time() - last_detection_time )> 10:
                        mode =0
                        state_initial_time = time.time()
                        fanqie_break_count +=1
                        play_interrupt()





            # 帧率
            img.draw_string(10, 210, "FPS: %0.2f" %(1000/t), scale=2, color=(0, 255, 0))
            # 工作模式（调试用）
            #img.draw_string(10, 180, "Mode: %d" %(mode), scale=2, color=(0, 255, 0))
            # 番茄数据
            if fanqie_total_count > 0:
                img.draw_string(10, 10, "Check: %d" %(fanqie_total_count), scale=2, color=(0, 255, 0))
            if fanqie_break_count > 0:
                img.draw_string(10, 40, "Break: %d" %(fanqie_break_count), scale=2, color=(255, 0, 0))
            # 右上角计时
            if mode ==0:
                img.draw_string(230, 10, "%s" %('25:00'), scale=3, color=(255, 255, 255))
            elif mode ==1:
                display_duration = time.time() - state_initial_time
                rest_time = fanqie_target_duration-display_duration
                # 将rest_time转换成分钟和秒
                rest_minute = int(rest_time/60)
                rest_second = int(rest_time%60)
                img.draw_string(230, 10, "%02d:%02d" %(rest_minute,rest_second), scale=3, color=(255, 0, 0))

            elif mode ==2:
                display_duration = time.time() - state_initial_time
                rest_time = reset_target_duration - display_duration
                # 将rest_time转换成分钟和秒
                rest_minute = int(rest_time/60)
                rest_second = int(rest_time%60)
                img.draw_string(230, 10, "%02d:%02d" %(rest_minute,rest_second), scale=3, color=(0, 255, 0))

            # 显示
            lcd.display(img)
    except Exception as e:
        raise e
    finally:
        kpu.deinit(task)

if __name__ == "__main__":
    try:
        # main program
        play_prepare()
        play_prepare()
        main()
    except Exception as e:
        sys.print_exception(e)
        lcd_show_except(e)
    finally:
        gc.collect()
