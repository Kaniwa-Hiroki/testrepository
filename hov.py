import time
from pymavlink import mavutil
import ctypes
import threading

class CustomThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}):
        threading.Thread.__init__(self, group=group, target=target, name=name)
        self.args = args
        self.kwargs = kwargs
        return
    
    def run(self):
        self._target(*self.args, **self.kwargs)

    def get_id(self):
        if hasattr(self, '_thread_id'):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id
    
    def raise_exception(self):
        thread_id = self.get_id()
        resu = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), ctypes.py_object(SystemExit))
        if resu > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), 0)
            print('Failure in raising exception')

def check_gpi(master):
    master.mav.request_data_stream_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_POSITION,
        1,
        1
    )

    while True:
        msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
        if msg is not None:
            print(msg)
        time.sleep(5)

def send_msg_rc(master, dict_rc):
    while True:
        master.mav.rc_channels_override_send(
            master.target_system, 
            master.target_component, 
            dict_rc["roll"], dict_rc["pitch"], dict_rc["throttle"], dict_rc["yaw"],
            0, 0, 0, 0) # R, P, Th, Y
        time.sleep(0.2)

def print_msg_AHRS2(master):
    msg = master.mav.command_long_encode(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE,
        0,  # confirmation
        mavutil.mavlink.MAVLINK_MSG_ID_AHRS2,
        0,0,0,0,0,0
    )
    master.mav.send(msg)
    msg = master.recv_match(type="AHRS2", blocking=True)
    print(msg)

def print_msg_DISTANCE_SENSOR(master):
    msg = master.mav.command_long_encode(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE,
        0,  # confirmation
        mavutil.mavlink.MAVLINK_MSG_ID_DISTANCE_SENSOR,
        0,0,0,0,0,0
    )

    master.mav.send(msg)
    res = master.recv_match(type="DISTANCE_SENSOR", blocking=True)
    #print(res.current_distance * 0.01)
    return res.current_distance * 0.01

def set_rc(dict_rc, th_send_msg_rc):
    if th_send_msg_rc is not None:
        th_send_msg_rc.raise_exception()
        th_send_msg_rc.join()
        th_send_msg_rc = None
    th_send_msg_rc = CustomThread(
        name="send_msg_rc", target=send_msg_rc, 
        args=(master, dict_rc),
        kwargs={})
    th_send_msg_rc.start()
    return th_send_msg_rc

class HovController():
    def __init__(self, current_alt, throttle=1430, min_throttle=1400, 
                 max_throttle=1460, value_t=20, error=0.1, width=1):
        self.current_alt = current_alt
        self.pre_alt = self.current_alt
        self.current_v = self.current_alt - self.pre_alt
        self.pre_v = self.current_v
        self.current_a = self.current_v - self.pre_v

        self.throttle = throttle
        self.min_throttle = min_throttle
        self.max_throttle = max_throttle
        self.value_t = value_t
        self.error = error
        self.width = width

    def get_throttle(self, current_alt, target_alt, error_coef_1=10, 
                     error_coef_2=4, width_coef_1=4, width_coef_2=2):
        self.pre_alt = self.current_alt
        self.current_alt = current_alt
        self.pre_v = self.current_v
        self.current_v = self.current_alt - self.pre_alt
        self.current_a = self.current_v - self.pre_v

        future_alt = self.current_alt + self.current_v * self.value_t + \
                     0.5 * self.current_a * self.value_t ** 2
        if future_alt > target_alt + self.error * error_coef_1:
            self.throttle -= self.width * width_coef_1
        elif future_alt > target_alt + self.error * error_coef_2:
            self.throttle -= self.width * width_coef_2
        elif future_alt > target_alt + self.error:
            self.throttle -= self.width
        elif future_alt < target_alt - self.error * error_coef_1:
            self.throttle += self.width * width_coef_1
        elif future_alt < target_alt - self.error * error_coef_2:
            self.throttle += self.width * width_coef_2
        elif future_alt < target_alt - self.error:
            self.throttle += self.width

        if self.throttle > self.max_throttle:
            self.throttle = self.max_throttle
        elif self.throttle < self.min_throttle:
            self.throttle = self.min_throttle
        return int(self.throttle)

# SITLへの接続
master = mavutil.mavlink_connection('tcp:127.0.0.1:5762')
master.wait_heartbeat()

th_send_msg_rc = None
dict_rc = {}
dict_rc["roll"] = 0
dict_rc["pitch"] = 0
dict_rc["throttle"] = 0
dict_rc["yaw"] = 0

# ARM
print("ARM")
while True:
    master.arducopter_arm()
    res = master.recv_match(type="COMMAND_ACK", blocking=True)
    if res.result == 0:
        break
    time.sleep(0.1)
start_alt = print_msg_DISTANCE_SENSOR(master)

# TAKEOFF & HOV
print("TAKEOFF & HOV")
current_alt = print_msg_DISTANCE_SENSOR(master)
hov = HovController(current_alt)
for i in range(200):
    current_alt = print_msg_DISTANCE_SENSOR(master)
    throttle = hov.get_throttle(current_alt, 2)
    dict_rc["throttle"] = throttle
    th_send_msg_rc = set_rc(dict_rc, th_send_msg_rc)
    print(throttle, current_alt)
    time.sleep(0.04)

print(f"HOV START {print_msg_DISTANCE_SENSOR(master)}")
time.sleep(10)
print(f"HOV_END {print_msg_DISTANCE_SENSOR(master)}")

master.set_mode(master.mode_mapping()["LAND"])
while current_alt > start_alt:
    current_alt = print_msg_DISTANCE_SENSOR(master)
    print(current_alt, start_alt)
    time.sleep(0.2)


# DISARM
print("DISARM")
dict_rc["throttle"] = 0
th_send_msg_rc = set_rc(dict_rc, th_send_msg_rc)
master.arducopter_disarm()

master.set_mode(master.mode_mapping()["STABILIZE"])

# finalize
if th_send_msg_rc is not None:
    th_send_msg_rc.raise_exception()
    th_send_msg_rc.join()

# 接続を閉じる
master.close()