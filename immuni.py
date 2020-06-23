import os
from sys import argv
from shutil import copyfile
import time
import datetime as dt

import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property
from robobrowser import RoboBrowser

import numpy as np
import pandas as pd
import scipy.stats as sps
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib import ticker

import schedule

DIR = os.path.dirname(os.path.realpath(__file__))

def highest_density_interval(pmf, p=.9, debug=False):
    # If we pass a DataFrame, just call this recursively on the columns
    if(isinstance(pmf, pd.DataFrame)):
        return pd.DataFrame([highest_density_interval(pmf[col], p=p) for col in pmf],
                            index=pmf.columns)
    
    cumsum = np.cumsum(pmf.values)
    
    # N x N matrix of total probability mass for each low, high
    total_p = cumsum - cumsum[:, None]
    
    # Return all indices with total_p > p
    lows, highs = (total_p > p).nonzero()
    
    # Find the smallest range (highest density)
    best = (highs - lows).argmin()
    
    low = pmf.index[lows[best]]
    high = pmf.index[highs[best]]
    
    return pd.Series([low, high],
                     index=[f'Low_{p*100:.0f}',
                            f'High_{p*100:.0f}'])


def gamma_params(x):
    omega = x / 864.4295302013422
    sigma = omega * 11.201342281879196
    rate = (omega + np.sqrt(omega**2 + 4*(sigma**2))) / (2*(sigma**2))
    shape = 1 + omega*rate
    scale = 1 / rate
    return (shape, scale)


def gamma_priors(x, sigma):
    omega = x
    rate = (omega + np.sqrt(omega**2 + 4*(sigma**2))) / (2*(sigma**2))
    shape = 1 + omega*rate
    scale = 1 / rate
    return (shape, scale)


def plot(results):
    fig, ax = plt.subplots(1,2,figsize=(14,5))

    ax[0].plot(
        results.index,
        results["ML"],
        label='Most Likely',
        c='k',                   
    )
    ax[0].fill_between(results.index,
                    results['Low_50'],
                    results['High_50'],
                    color='k',
                    alpha=.1,
                    lw=0,
                    label='HDI 50%')
    ax[0].fill_between(results.index,
                    results['Low_95'],
                    results['High_95'],
                    color='k',
                    alpha=.05,
                    lw=0,
                    label='HDI 95%')
    ax[0].legend(loc="upper left")
    for yy in np.arange(1e6, 10e6+1, 1e6):
        ax[0].axhline(yy, c="r", ls=":", alpha=.25)
    
    ax[0].set_ylim(0, results['High_95'][-1])

    ax[1].plot(
        results.index,
        results["ML"]/it_pop*100,
        label='Most Likely',
        c='k',                   
    )
    ax[1].fill_between(results.index,
                    results['Low_50']/it_pop*100,
                    results['High_50']/it_pop*100,
                    color='k',
                    alpha=.1,
                    lw=0,
                    label='HDI 50%')
    ax[1].fill_between(results.index,
                    results['Low_95']/it_pop*100,
                    results['High_95']/it_pop*100,
                    color='k',
                    alpha=.05,
                    lw=0,
                    label='HDI 95%')
    ax[1].legend(loc="upper left")
    for yy in np.arange(2, 20+1, 2):
        ax[1].axhline(yy, c="r", ls=":", alpha=.25)

    ax[1].set_ylim(0, results['High_95'][-1]/it_pop*100)

    ax[0].set_title("Downloads totali stimati dell'app Immuni")
    ax[0].set_xlim(results.index[0], results.index[-1])
    ax[0].xaxis.set_major_locator(mdates.WeekdayLocator())
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax[0].xaxis.set_minor_locator(mdates.DayLocator())
    ax[0].set_ylabel("Milioni di downloads")

    ax[1].set_title("Percentuale di downloads stimati su popolazione")
    ax[1].set_xlim(results.index[0], results.index[-1])
    ax[1].xaxis.set_major_locator(mdates.WeekdayLocator())
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax[1].xaxis.set_minor_locator(mdates.DayLocator())
    ax[1].yaxis.set_major_formatter(ticker.PercentFormatter())
    ax[1].set_ylabel("Percentuali")

    fig.set_facecolor('w')
    plt.savefig("immuni.png", bbox_inches='tight')


def bayes(reviews):
    D_rows = sps.gamma.pdf(reviews["reviews"].values[:,None], a=shape, scale=scale)
    D_rows = D_rows.transpose()
    likelihoods = pd.DataFrame(
        data=D_rows,
        columns=reviews.index,
        index=D_range[1:]
    )
    posteriors = pd.DataFrame(
        index=D_range[1:],
        columns=reviews.index,
        data={
            reviews.index[0]: prior0,
        }
    )

    for previous_day, current_day in zip(reviews.index[:-1], reviews.index[1:]):
        current_prior = priors @ posteriors[previous_day]
        numerator = likelihoods[current_day] * current_prior
        denominator = np.sum(numerator)
        posteriors[current_day] = numerator/denominator
    
    return posteriors


def update():
    print(f"{dt.datetime.now()} Updating reviews...")
    browser.open("https://play.google.com/store/apps/details?id=it.ministerodellasalute.immuni&hl=it_IT")
    google = browser.select(".AYi5wd.TBRnV")
    google_reviews = int(google[0].text.replace(".", ""))

    print(f"{dt.datetime.now()} Google R: {google_reviews}")

    browser.open("https://apps.apple.com/it/app/immuni/id1513940977")
    apple = browser.select(".we-customer-ratings__count.small-hide.medium-show")
    apple_reviews = int(apple[0].text.split(" ")[0].replace(",", ""))

    print(f"{dt.datetime.now()} Apple R: {apple_reviews}")

    total_reviews = google_reviews + apple_reviews

    today = pd.Timestamp.now().date()

    dic = {
        "date": [today],
        "google_reviews": [google_reviews],
        "apple_reviews": [apple_reviews],
        "reviews": [total_reviews],
    }

    old = pd.read_csv("immuni-reviews.csv", index_col=["date"], parse_dates=["date"])

    df = pd.DataFrame(dic)
    df.set_index("date", inplace=True)
    df = pd.concat([old, df], ignore_index=False)
    df.to_csv("immuni-reviews.csv", index=True)

    print(f"{dt.datetime.now()} DONE")

    print(f"{dt.datetime.now()} Calculating posteriors...")
    posteriors = bayes(df)
    print(f"{dt.datetime.now()} Done")

    print(f"{dt.datetime.now()} Extracting ML and HDI...")
    most_likely_values = posteriors.idxmax(axis=0).rename('ML')
    hdi50 = highest_density_interval(posteriors, p=.5)
    hdi95 = highest_density_interval(posteriors, p=.95)
    results = pd.concat([most_likely_values, hdi50, hdi95], axis=1)
    print(f"{dt.datetime.now()} Done")

    print(f"{dt.datetime.now()} Plotting...")
    plot(results)
    copyfile("immuni.png", f"{DIR}/../covid-19-jupyter/immuni.png")
    print(f"{dt.datetime.now()} DONE!")


it_pop = 60359546
steps = int(6e3)

D_max = 6e7
D_range = np.linspace(0, D_max, steps+1)
R_max = 3e6
R_range = np.linspace(0, R_max, steps+1)

browser = RoboBrowser(history=False, parser="lxml")

print(f"{dt.datetime.now()} Generating priors...")

prior_shape, prior_scale = gamma_priors(D_range[1:], 1e6)
priors = sps.gamma(
    a=prior_shape,
    scale=prior_scale
).pdf(D_range[1:, None])

priors /= priors.sum(axis=0)
print(f"{dt.datetime.now()} Done")

prior0 = np.ones_like(D_range[1:])/len(D_range[1:])
prior0 /= prior0.sum()

print(f"{dt.datetime.now()} Calculating likelihoods...")
shape, scale = gamma_params(D_range[1:])
print(f"{dt.datetime.now()} Done")

if len(argv) > 1:
    update()
schedule.every().day.at("07:00").do(update)

while True:
   schedule.run_pending()
   time.sleep(1)
