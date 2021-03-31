import os
import pickle
import pandas as pd
from rlenv.StockTradingEnv0 import StockTradingEnv
import baostock as bs
import datetime
from stable_baselines import PPO2


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
            for i in range(len(df_code) - 1):
                action, _states = model.predict(obs)
                obs, rewards, done, info = env.step(action)
                profit = env.render()
                day_profits.append(profit)
                if done:
                    break
        return day_profits

