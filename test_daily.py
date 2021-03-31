import os
import pickle
import pandas as pd
from rlenv.StockTradingEnv0 import StockTradingEnv
from stable_baselines.common.vec_env import DummyVecEnv
import baostock as bs
import datetime
import json
import logging
from stable_baselines import PPO2
import schedule
import time

logging.basicConfig(level=logging.DEBUG,#控制台打印的日志级别
   filename='test.log',
   filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
   #a是追加模式，默认如果不写的话，就是追加模式
   format=
   '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
   #日志格式
   )

class DailyTester(object):
    def __init__(self, date_start, date_end):
        self._bs = bs
        bs.login()
        self.fields = "date,code,open,high,low,close,volume,amount," \
                      "adjustflag,turn,tradestatus,pctChg,peTTM," \
                      "pbMRQ,psTTM,pcfNcfTTM,isST"
        # self.date_start = datetime.datetime.now().strftime("%Y-%m-%D")
        self.date_start = date_start
        self.date_end = date_end
        if not os.path.exist("./stockdata/status"):
            os.mkdir("./stockdata/status")

    def exit(self):
        bs.logout()

    def get_single_code(self, code):
        df_code = bs.query_history_k_data_plus(code, self.fields,
                                                   start_date=self.date_start, end_date = self.date_end).get_data()
        return df_code

    def test(self, code):
        df_code = self.get_single_code(code)
        if os.path.exists("./models/"+code):
            model = model = PPO2.load("./models/"+code)
            env = DummyVecEnv([lambda: StockTradingEnv(df_code)])
            if os.path.exists("./stockdata/status/"+code):
                with open("./stockdata/status/"+code,"r") as f:
                    status = json.load(fp = f)
                    obs = env.reset()
                    env.set_status(status)
            else:
                obs = env.reset()
            day_profits = []
            for i in range(len(df_code) - 1):
                action, _states = model.predict(obs)
                obs, rewards, done, info = env.step(action)
                profit = env.render()
                day_profits.append(profit)
                if done:
                    with open("./stockdata/status/"+code,"w+") as f:
                        json.dump(env.get_status(), fp = f)
                    break
        return day_profits

def get_code_list():
    code_list_file = "./config/code_test.txt"
    with open(code_list_file, 'r') as f:
        lines = f.readlines()
        code_list = []
        for l in lines:
            code_list.append(l.strip())
        return code_list
    return []
def main():
    code_list = get_code_list()
    data_end = datetime.datetime.now()
    data_end_str = data_end.strftime("%Y-%m-%d")
    data_start = data_end-datetime.timedelta(days=1)
    data_start_str = data_start.strftime("Y-%m-%d")
    dt = DailyTester(data_start_str, data_end_str)
    for code in code_list:
        day_profits = dt.test(code)
        logging.info(str(code)+","+data_end_str+","+str(day_profits))

if __name__ == "__main__":
    schedule.every().day.at("19:30").do(main)
    while True:
        schedule.run_pending()
        time.sleep(1)