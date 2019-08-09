Q-Learning Bubble Shooter bot
=============================

Bubble Shooter game: [https://www.kongregate.com/games/Paulussss/bubbles-shooter](https://www.kongregate.com/games/Paulussss/bubbles-shooter)

Demo gameplay:

[![Basic Puzlogic game bot with OpenCV and Python](https://img.youtube.com/vi/9K343IWO2N4/0.jpg)](https://www.youtube.com/watch?v=9K343IWO2N4)

A blog post writeup on how and why the bot was built: [Q-learning bot for Bubble Shooter](https://tautvidas.com/blog/2019/08/q-learning-bot-for-bubble-shooter/)

Requires:

- `geckodriver` in your `$PATH`
- `OpenCV` installed on the system

```
virtualenv -p python3 --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt
cd webpage && sudo python -m SimpleHTTPServer 80 # to run a local version of Bubble Shooter
python train.py
python demo.py
```
