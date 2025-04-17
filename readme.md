## How to use
### Setting enviroment
```bash
$ pip install -r requirements.txt
```

### Pipeline: Get News data(by RSS) -> Preprocessing -> Topic modeling
```bash
$ python run.py
```

### Discord Bot
```bash
$ nohup python -u src/discord/bot.py > logs/discord_bot.log 2>&1 &
```