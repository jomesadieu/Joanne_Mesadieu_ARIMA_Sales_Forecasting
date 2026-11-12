"""Reproduce supplied ARIMA(5,1,0) walk-forward method on both sales files."""
from pathlib import Path
import json, warnings, platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA
import statsmodels
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
ORDER=(5,1,0)
TRAIN_FRACTION=.66

def load_sales(name):
    """Parse source dates explicitly; reject missing, duplicate or gapped months."""
    if name=='shampoo':
        frame=pd.read_csv(ROOT/'data/shampoo.csv')
        # The teaching file has relative years 1–3. Preserve its original
        # parser's display convention (1901–1903), not asserted historical dates.
        dates=pd.to_datetime('190'+frame['Month'].astype(str),format='%Y-%m')
        values=pd.to_numeric(frame['Sales'],errors='raise')
    else:
        frame=pd.read_csv(ROOT/'data/BeerWineLiquor.csv')
        dates=pd.to_datetime(frame['date'],format='%m/%d/%Y')
        values=pd.to_numeric(frame['beer'],errors='raise')
    series=pd.Series(values.to_numpy(dtype=float),index=pd.DatetimeIndex(dates),name='sales').sort_index()
    if series.index.has_duplicates: raise ValueError('Duplicate dates in '+name)
    series=series.asfreq('MS')
    if series.isna().any() or not np.isfinite(series.to_numpy()).all():
        raise ValueError('Missing/nonfinite observations or missing months in '+name)
    return series

def evaluate_forecasts(actual,predicted):
    a,p=np.asarray(actual,dtype=float),np.asarray(predicted,dtype=float)
    if a.shape!=p.shape or len(a)==0: raise ValueError('Invalid forecast alignment')
    error=a-p
    nonzero=a!=0
    return dict(MAE=float(np.abs(error).mean()),RMSE=float(np.sqrt(np.mean(error**2))),
                MAPE_percent=float(np.mean(np.abs(error[nonzero]/a[nonzero]))*100) if nonzero.any() else None)

def walk_forward_forecast(series):
    """Fit, predict next month, then reveal that month's actual observation."""
    split=int(len(series)*TRAIN_FRACTION)
    train,test=series.iloc[:split],series.iloc[split:]
    history=train.to_numpy().tolist(); records=[]; warning_rows=[]
    for date,actual in test.items():
        # Fixed order was supplied by the instructor; no hold-out tuning.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fitted=ARIMA(np.asarray(history,dtype=float),order=ORDER).fit()
        fc=fitted.get_forecast(steps=1)
        mean=float(np.asarray(fc.predicted_mean)[0])
        lower,upper=np.asarray(fc.conf_int(alpha=.05))[0]
        records.append(dict(date=date,actual=float(actual),predicted=mean,
                            lower_95=float(lower),upper_95=float(upper),
                            naive_last_month=float(history[-1]),
                            seasonal_naive_12=float(history[-12]),training_observations=len(history),
                            optimizer_converged=bool(fitted.mle_retvals.get('converged',True))))
        for w in caught:
            warning_rows.append(dict(date=str(date.date()),category=w.category.__name__,message=str(w.message)))
        history.append(float(actual))
    forecasts=pd.DataFrame(records).set_index('date')
    assert forecasts.index.equals(test.index)
    assert forecasts.training_observations.to_list()==list(range(split,len(series)))
    return train,test,forecasts,warning_rows

def plot_actual_vs_predicted(name,series,train,forecasts,metrics):
    title='Shampoo' if name=='shampoo' else 'Beer (BeerWineLiquor.csv, beer column)'
    fig,axes=plt.subplots(2,1,figsize=(13,9),gridspec_kw={'height_ratios':[1,1.2]},layout='constrained')
    ax=axes[0]
    ax.plot(series.index,series.values,color='#143d59',lw=1.6,label='Actual sales')
    ax.plot(forecasts.index,forecasts.predicted,color='#e76f51',lw=1.5,ls='--',label='One-step ARIMA predictions')
    ax.axvline(forecasts.index[0],color='#555',ls=':',label='Start of test period')
    ax.axvspan(forecasts.index[0],forecasts.index[-1],color='#e9c46a',alpha=.12)
    ax.set_title('Full series and chronological train/test split',fontsize=12)
    ax.set_ylabel('Sales (source units)'); ax.legend(loc='upper left',fontsize=9,ncol=2);ax.grid(alpha=.18)
    ax=axes[1]
    ax.plot(forecasts.index,forecasts.actual,color='#143d59',lw=2,marker='o',ms=3,label='Actual test sales')
    ax.plot(forecasts.index,forecasts.predicted,color='#e76f51',lw=1.8,ls='--',marker='s',ms=3,label='Predicted sales')
    ax.fill_between(forecasts.index,forecasts.lower_95,forecasts.upper_95,color='#e76f51',alpha=.15,label='95% one-step prediction interval')
    ax.set_title(f'Test-period comparison | RMSE = {metrics["RMSE"]:.2f} | MAE = {metrics["MAE"]:.2f}',fontsize=12)
    ax.set(xlabel='Month',ylabel='Sales (source units)');ax.legend(loc='upper left',fontsize=9);ax.grid(alpha=.18)
    for ax in axes:
        locator=mdates.AutoDateLocator(minticks=5,maxticks=10)
        ax.xaxis.set_major_locator(locator);ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    subtitle=f'ARIMA{ORDER} | {len(train)} initial training months, {len(forecasts)} walk-forward test months'
    if name=='shampoo': subtitle+='\n1901–1903 are display placeholders for source years 1–3.'
    fig.suptitle(title+' Sales: Actual vs Predicted\n'+subtitle,fontsize=16,fontweight='bold')
    fig.savefig(ROOT/f'figures/{name}_arima_actual_vs_predicted.png',dpi=300,bbox_inches='tight');plt.close(fig)

def main():
    for folder in ['figures','results']: (ROOT/folder).mkdir(exist_ok=True,parents=True)
    metric_rows=[]; details={}; all_warnings=[]
    for name in ['shampoo','beer']:
        series=load_sales(name)
        print(f'Forecasting {name}: {len(series)} monthly observations',flush=True)
        train,test,forecasts,warning_rows=walk_forward_forecast(series)
        forecasts.to_csv(ROOT/f'results/{name}_actual_vs_predicted.csv',date_format='%Y-%m-%d')
        for model,col in [('ARIMA(5,1,0)','predicted'),('Naive: last month','naive_last_month'),('Seasonal naive: last year','seasonal_naive_12')]:
            metric_rows.append(dict(dataset=name,model=model,train_months=len(train),test_months=len(test),
                                    **evaluate_forecasts(forecasts.actual,forecasts[col])))
        arima_metrics=evaluate_forecasts(forecasts.actual,forecasts.predicted)
        plot_actual_vs_predicted(name,series,train,forecasts,arima_metrics)
        details[name]=dict(total_months=len(series),train_months=len(train),test_months=len(test),
                           start=str(series.index[0].date()),end=str(series.index[-1].date()),
                           test_start=str(test.index[0].date()),test_end=str(test.index[-1].date()),
                           interval_coverage=float(((forecasts.actual>=forecasts.lower_95)&(forecasts.actual<=forecasts.upper_95)).mean()),
                           failed_convergence=int((~forecasts.optimizer_converged).sum()))
        all_warnings.extend(dict(dataset=name,**row) for row in warning_rows)
        print(f'{name}: RMSE {arima_metrics["RMSE"]:.3f}; MAE {arima_metrics["MAE"]:.3f}',flush=True)
    metrics=pd.DataFrame(metric_rows);metrics.to_csv(ROOT/'results/forecast_metrics.csv',index=False)
    pd.DataFrame(all_warnings,columns=['dataset','date','category','message']).to_csv(ROOT/'results/fit_warnings.csv',index=False)
    details['configuration']=dict(order=list(ORDER),train_fraction=TRAIN_FRACTION,
        method='Expanding-window, one-step-ahead, refit at every test month',
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,statsmodels=statsmodels.__version__,matplotlib=matplotlib.__version__))
    (ROOT/'results/run_summary.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
    write_readme(metrics,details)
    print(metrics.round(3).to_string(index=False),flush=True)
    return metrics

def write_readme(metrics,details):
    rows=['| Dataset | Model | Train months | Test months | MAE | RMSE | MAPE (%) |',
          '|---|---|---:|---:|---:|---:|---:|']
    for _,r in metrics.iterrows():
        rows.append(f'| {r.dataset.title()} | {r.model} | {r.train_months} | {r.test_months} | {r.MAE:.3f} | {r.RMSE:.3f} | {r.MAPE_percent:.2f} |')
    interpretation=[]
    for name in ['shampoo','beer']:
        t=metrics[metrics.dataset==name];a=t[t.model=='ARIMA(5,1,0)'].iloc[0];best=t.loc[t.RMSE.idxmin()]
        interpretation.append(f'{name.title()}: ARIMA RMSE is {a.RMSE:.3f}. The lowest-RMSE method among these three fixed methods is {best.model} ({best.RMSE:.3f}).')
    text='\n\n'.join(['# ARIMA Sales Forecasting — Shampoo and Beer','Joanne Mesadieu',
      'Forecast monthly shampoo and beer sales using the supplied ARIMA notebook and upload plots comparing predictions with actual sales.',
      '## Required plots','![Shampoo actual vs predicted](figures/shampoo_arima_actual_vs_predicted.png)\n\n![Beer actual vs predicted](figures/beer_arima_actual_vs_predicted.png)',
      '## Data',
      'The supplied shampoo.csv contains 36 monthly sales observations. Relative source labels 1-01 through 3-12 are displayed as 1901-01 through 1903-12 to match the supplied notebook. These are plotting placeholders, not independently verified calendar years. BeerWineLiquor.csv contains 324 monthly observations from January 1992 through December 2018 in its beer column. The filename is retained; its contents have not been independently verified as beer-only revenue. Sales units are not stated in the uploads, so figures use source units. Both series are checked for missing values, duplicate dates and missing months; no values are imputed.',
      '## Forecasting method',
      'Both series use statsmodels.tsa.arima.model.ARIMA with order=(5,1,0), preserving the supplied example. p=5 uses five autoregressive lags, d=1 differences the series once, and q=0 has no moving-average term. The first int(0.66 × N) observations form the initial training set: 23 shampoo months and 213 beer months. The remaining 13 and 111 months form the test periods. No random shuffle is used.',
      'For every test month: fit ARIMA on the expanding history, predict exactly the next month, save its 95% prediction interval, then add the newly observed actual sale to history. Test observations become available only after their own prediction is made. This is one-step walk-forward evaluation, not a single forecast of the entire test horizon from the original training cutoff. No parameters are tuned using the test results.',
      '## Results','\n'.join(rows),'\n\n'.join(interpretation),
      'Metrics are calculated automatically from aligned test-date CSVs. MAE is mean absolute error; RMSE is the square root of mean squared error; MAPE is mean absolute percentage error. Lower is better. Naive and seasonal-naive benchmarks forecast the previous month and the same month one year earlier, respectively, using only available history.',
      '## Interpretation and limitations',
      'The beer series has recurring annual peaks, particularly near year end. A nonseasonal ARIMA(5,1,0) need not reproduce these well; a seasonal ARIMA model with period 12 would be a useful follow-up. The task’s fixed ARIMA specification is retained instead of selecting a different model after viewing the test outcomes. Shampoo has only three years of data, making estimation and evaluation sensitive to a few observations. Prediction intervals depend on model assumptions and are not guarantees. Errors are in each dataset’s original scale and should not be directly compared across datasets.',
      f'Fits not reporting optimizer convergence: shampoo {details["shampoo"]["failed_convergence"]}, beer {details["beer"]["failed_convergence"]}. Full warnings are saved in results/fit_warnings.csv.',
      '## Run',
      'Use Python 3.12. From the repository root:\n```bash\npython -m pip install -r requirements.txt\npython src/arima_forecasting.py\n```\nOr run all cells in ARIMA_Sales_Forecasting.ipynb. The notebook is self-contained: it includes the small supplied CSVs and executable analysis source, creates the project folders if necessary, and writes every figure and CSV. The original teaching notebook is preserved in reference/ARIMA_original.ipynb.',
      '## Submission',
      'Publish this project as a public GitHub repository named Joanne_Mesadieu_ARIMA_Sales_Forecasting. Submit the actual repository URL and upload figures/shampoo_arima_actual_vs_predicted.png and figures/beer_arima_actual_vs_predicted.png to the assignment. A repository link is valid only after it has actually been published; no URL is fabricated in this project.',
      '## Software versions','```json\n'+json.dumps(details['configuration']['versions'],indent=2)+'\n```',
      '## References',
      'Inputs: user-provided shampoo.csv, BeerWineLiquor.csv and ARIMA.ipynb. Documentation: https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html and https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMAResults.get_forecast.html'])
    (ROOT/'README.md').write_text(text,encoding='utf-8')

if __name__=='__main__':
    with threadpool_limits(limits=1): main()
