import time

import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property
from robobrowser import RoboBrowser

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib import ticker

import schedule

it_pop = 60359546
downloads_reviews_rates = pd.read_csv("rates_levels.csv")["rates"]

browser = RoboBrowser(history=False, parser="lxml")


def plot(df, alpha1=.05, alpha2=.1):
    fig, ax = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    ax[0].fill_between(
        df.index,
        df["total_min_2"],
        df["total_max_2"],
        alpha=.1, label=fr"$\pm {alpha2*100:.0f}\%$"
    )
    ax[0].fill_between(
        df.index,
        df["total_min_1"],
        df["total_max_1"],
        alpha=.2, label=fr"$\pm {alpha1*100:.0f}\%$"
    )
    ax[0].plot(
        df.index,
        df["total_mean"],
        c="k", label="media"
    )
    # PERCENTAGE
    ax[1].plot(
        df.index,
        df["total_mean"]/it_pop*100,
        c="k", label="media"
    )
    ax[1].fill_between(
        df.index,
        df["total_min_2"]/it_pop*100,
        df["total_max_2"]/it_pop*100,
        alpha=.1, label=fr"$\pm {alpha2*100:.0f}\%$"
    )
    ax[1].fill_between(
        df.index,
        df["total_min_1"]/it_pop*100,
        df["total_max_1"]/it_pop*100,
        alpha=.2, label=fr"$\pm {alpha1*100:.0f}\%$"
    )
    ax[0].set_title("Stima downloads totali (Google Play + App Store)")
    ax[0].ticklabel_format(style='plain', axis="y")
    ax[0].axhline(0, c="k", alpha=.2)
    ax[0].xaxis.set_major_locator(mdates.WeekdayLocator())
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax[0].xaxis.set_minor_locator(mdates.DayLocator())
    ax[0].legend(loc="upper left")

    ax[1].set_title("Stima percentuale di downloads su popolazione italiana")
    ax[1].yaxis.set_major_formatter(ticker.PercentFormatter())
    ax[1].axhline(0, c="k", alpha=.2)
    ax[1].legend(loc="upper left")

    fig.set_facecolor('w')
    plt.savefig("immuni.png", bbox_inches='tight')


def update():
    print("-> UPDATING")
    browser.open("https://play.google.com/store/apps/details?id=it.ministerodellasalute.immuni&hl=it_IT")
    google = browser.select(".AYi5wd.TBRnV")
    google_reviews = int(google[0].text.replace(".", ""))
    google_downloads = google_reviews * 1 / downloads_reviews_rates

    browser.open("https://apps.apple.com/it/app/immuni/id1513940977")
    apple = browser.select(".we-customer-ratings__count.small-hide.medium-show")
    apple_reviews = int(apple[0].text.split(" ")[0].replace(",", ""))
    apple_downloads = apple_reviews * 1 / downloads_reviews_rates

    total_downloads = google_downloads + apple_downloads
    # percentage = total_downloads / it_pop

    today = pd.Timestamp.now().date()

    dic = {
        "date": [today],
        "total_min_2": [total_downloads[0]],
        "total_min_1": [total_downloads[1]],
        "total_mean": [total_downloads[2]],
        "total_max_1": [total_downloads[3]],
        "total_max_2": [total_downloads[4]],
        "google_reviews": [google_reviews],
    }

    old = pd.read_csv("immuni.csv", index_col=["date"], parse_dates=["date"])

    df = pd.DataFrame(dic)
    df.set_index("date", inplace=True)
    df = pd.concat([old, df], ignore_index=False)
    df.to_csv("immuni.csv", index=True)

    print("<- DONE")
    print("-> Plotting")
    plot(df=df)
    print("<- Done")


# update()
schedule.every().day.at("09:00").do(update)

while True:
    schedule.run_pending()
    time.sleep(1)
