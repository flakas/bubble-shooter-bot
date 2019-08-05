Q-Learning Bubble Shooter bot
=============================

Bubble Shooter game: [https://www.kongregate.com/games/Paulussss/bubbles-shooter](https://www.kongregate.com/games/Paulussss/bubbles-shooter)

Requires:

- `geckodriver` in your `$PATH`
- `OpenCV` installed on the system

```
virtualenv -p python3 --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt
python train_multiprocess.py
python demo.py
```
