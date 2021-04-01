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

logging.basicConfig(level=logging.ERROR,#控制台打印的日志级别
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
        self.float_fields = ['open','high','low','close','volume','amount','adjustflag','turn','tradestatus','pctChg','peTTM','pbMRQ','psTTM','pcfNcfTTM','isST']
        self.status_keys = ['balance','net_worth','max_net_worth','shares_held','cost_basis','total_shares_sold','total_sales_value']
        self.date_start = date_start
        self.date_end = date_end
        if not os.path.exists("./stockdata/status"):
            os.mkdir("./stockdata/status")

    def exit(self):
        bs.logout()

    def find_file(self, path, name):
        # print(path, name)
        for root, dirs, files in os.walk(path):
            for fname in files:
                if name in fname:
                    return os.path.join(root, fname)

    def get_single_code(self, code):
        df_code = bs.query_history_k_data_plus(code, self.fields, start_date=self.date_start, end_date = self.date_end).get_data()
        for ff in self.float_fields:
            df_code[ff] = df_code[ff].astype('float64')

        return df_code
    def get_single_code_file(self, code):
        stock_file = self.find_file('./stockdata/test', str(code))
        df = pd.read_csv(stock_file)
        return df

    def set_status(self, env, status):
        for k,v in status.items():
            env.set_attr(k, v)
    def get_status(self, env, keys):
        res = {}
        for k in keys:
            res[k] = env.get_attr(k)[0]
        return res

    def test(self, code):
        df_code = self.get_single_code(code)
        print(df_code)
        day_profits = []
        if os.path.exists("./models/"+code):
            model = PPO2.load("./models/"+code)
            env = DummyVecEnv([lambda: StockTradingEnv(df_code)])
            if os.path.exists("./stockdata/status/"+code):
                with open("./stockdata/status/"+code,"r") as f:
                    status = json.load(fp = f)
                    obs = env.reset()
                    self.set_status(env, status)
            else:
                obs = env.reset()
            
            for i in range(len(df_code) - 1):
                action, _states = model.predict(obs)
                obs, rewards, done, info = env.step(action)
                print("-"*10)
                print(action)
                print("-"*10)
                profit = env.render()
                day_profits.append(profit)
                if done:
                    break
            with open("./stockdata/status/"+code,"w+") as f:
                json.dump(self.get_status(env, self.status_keys), fp = f)
        else:
            print("no models!")        
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
    print(code_list)
    data_end = datetime.datetime.now()
    data_end_str = data_end.strftime("%Y-%m-%d")
    data_start = data_end-datetime.timedelta(days=2)
    data_start_str = data_start.strftime("%Y-%m-%d")
    print(data_end_str, data_start_str)
    dt = DailyTester(data_start_str, data_end_str)
    for code in code_list:
        day_profits = dt.test(code)
        logging.error(str(code)+","+data_end_str+","+str(day_profits))

if __name__ == "__main__":
    schedule.every().day.at("19:30").do(main)
    while True:
        schedule.run_pending()
        time.sleep(1)
    # main()