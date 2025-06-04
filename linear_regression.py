import statsmodels.api as sm
import statsmodels.formula.api as smf

dta = sm.datasets.get_rdataset("Guerry", "HistData", cache=True)
df = dta.data[["Lottery", "Literacy", "Wealth", "Region"]].dropna()
display(df.head())

mod = smf.ols(formula="Lottery ~ Literacy + Wealth + Region", data=df) 
res = mod.fit() 
print(res.summary()) 


