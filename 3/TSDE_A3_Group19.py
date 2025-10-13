import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import gaussian_kde
from statsmodels.tsa.stattools import adfuller

'''
Part I: Spurious Regression and Unit Roots
'''
print('\nPart I: Spurious Regression and Unit Roots')

# Question 1

# calculation for estimated beta by OLS regression
def getEstimatedBeta(input_matrix_X, input_vector_y):
    # calculate the inverse matrix (X′ * X)^{−1}
    df = input_matrix_X.T @ input_matrix_X
    df_inv = pd.DataFrame(np.linalg.inv(df.values),
                      index=df.columns,   # rows = original columns
                      columns=df.index) 
    
    # calculate estimated beta by (X′ * X)^{−1} * X′* y
    result_beta = df_inv @ input_matrix_X.T @ input_vector_y
    
    return result_beta

def runRegressionModel(input_y, input_x):
    # convert y to matrix y
    vector_y = pd.DataFrame(input_y)
    # print(vector_y)
    # convert y to matrix X
    matrix_X = pd.DataFrame({'intercept': 1, 'x': input_x})
    
    # calculate estimated beta by (X′ * X)^{−1} * X′* y
    estimate_beta = getEstimatedBeta(matrix_X, vector_y)

    # calculate estimated y by X * (estimated beta)
    estimate_y = matrix_X @ estimate_beta
    
    # calculate the residuals
    estimate_residuals = vector_y - estimate_y

    return matrix_X, estimate_beta, estimate_y, estimate_residuals

def runRegressionModel_ECM(input_y, input_x, input_z, input_p, input_q):
    # convert y to matrix y
    vector_y = pd.DataFrame(input_y[max(input_p, input_q):])
    vector_y = vector_y.reset_index(drop=True)
    # convert y to matrix X
    matrix_X = generateMatrixX_ECM(input_y, input_x, input_z, input_p, input_q)
    
    # calculate estimated beta by (X′ * X)^{−1} * X′* y
    estimate_beta = getEstimatedBeta(matrix_X, vector_y)

    # calculate estimated y by X * (estimated beta)
    estimate_y = matrix_X @ estimate_beta
    
    # calculate the residuals
    residuals = vector_y - estimate_y

    return matrix_X, estimate_beta, estimate_y, residuals

def runRegressionModel_ADF(input_delta_y, input_y, input_p):
    # convert y to matrix y
    vector_y = pd.DataFrame(input_delta_y[input_p:])
    vector_y = vector_y.reset_index(drop=True)
    # convert y to matrix X
    matrix_X = generateMatrixX_ADF(input_delta_y, input_y, input_p)
    
    # calculate estimated beta by (X′ * X)^{−1} * X′* y
    estimate_beta = getEstimatedBeta(matrix_X, vector_y)

    # calculate estimated y by X * (estimated beta)
    estimate_y = matrix_X @ estimate_beta
    
    # calculate the residuals
    estimate_residuals = vector_y - estimate_y

    return matrix_X, estimate_beta, estimate_y, estimate_residuals

def getStandardError(input_matrix, input_residuals):
    # get the shape of the matrix
    n, k = input_matrix.shape

    # estimate the error variance
    estimated_variance = (input_residuals.T @ input_residuals).iat[0, 0] / (n - k)
    
    df = input_matrix.T @ input_matrix
    df_inv = pd.DataFrame(np.linalg.inv(df.values),
                        index=df.columns,   # rows = original columns
                        columns=df.index) 

    # compute the variance-covariance matrix
    varCovMatrix = estimated_variance * df_inv

    # get the standard errors which are the diagonal element
    estimated_standard_error = np.sqrt(np.diag(varCovMatrix))
    
    return estimated_standard_error

def getRSquare(input_residuals_list, true_y_list):
    mean_y = np.average(true_y_list)
    ssr = 0.0
    sst = 0.0
    
    for res, true_y in zip(input_residuals_list, true_y_list):
        ssr += res ** 2
        sst += (true_y - mean_y) ** 2

    rSquare = 1 - (ssr / sst)
    
    return rSquare

def getKernelDensity(input_variables):
    kde = gaussian_kde(np.asarray(input_variables))
    x = np.linspace(min(input_variables), max(input_variables), 200)
    density = kde(x)
    
    return x, density
    
def distributionPlot(input_x_list, input_density_list, input_numberOfSimulations_list, input_xLabel, input_title, input_fileName):
    plt.figure(figsize=(10,6)) 
    
    for i in range(0,len(input_numberOfSimulations_list)):
        plt.plot(input_x_list[i], input_density_list[i], 
                 linestyle='-', linewidth=2,
                 label=f'T={input_numberOfSimulations_list[i]}')
    
    plt.title(f'{input_title}', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel(input_xLabel)
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(f'../3/figures/{input_fileName}.jpeg', dpi=300)
    plt.show()
    
def monteCarlo(input_B, input_t):
    this_beta_list = []
    this_t_list = []
    this_RSquare_list = []
    
    for i in range(0, input_B):
        v_t = np.random.normal(loc=0, scale=1, size=input_t+1)
        w_t = np.random.normal(loc=0, scale=1, size=input_t+1)
    
        this_y_list = []
        this_y_list.append(0)
        this_x_list = []
        this_x_list.append(0)

        for j in range(1, input_t+1):
            y_t = this_y_list[j-1] + v_t[j]
            this_y_list.append(y_t)
            x_t = this_x_list[j-1] + w_t[j]
            this_x_list.append(x_t)
        
        matrix_X, estimate_beta, hat_y, hat_residuals = runRegressionModel(this_y_list, this_x_list)
        
        estimate_SE = getStandardError(matrix_X, hat_residuals)
        
        t_statistic = estimate_beta.iat[1,0] / estimate_SE[1]
        
        rSquare = getRSquare(hat_residuals.squeeze(), this_y_list)
        
        this_beta_list.append(estimate_beta.iat[1, 0])
        this_t_list.append(t_statistic)
        this_RSquare_list.append(rSquare)
    
    return this_beta_list, this_t_list, this_RSquare_list
    
def partI_question1():
    time_list = [100, 500, 1000]
    numberOfSimulations_B = 5000

    beta_x_list = []
    beta_density_list = []
    t_x_list = [] 
    t_density_list = []
    rSquare_x_list = []
    rSquare_density_list = []

    for time in time_list:
        this_beta_list, this_t_list, this_RSquare_list = monteCarlo(numberOfSimulations_B, time)
            
        beta_x, beta_density = getKernelDensity(this_beta_list)
        beta_x_list.append(beta_x)
        beta_density_list.append(beta_density)
        
        t_x, t_density = getKernelDensity(this_t_list)
        t_x_list.append(t_x)
        t_density_list.append(t_density)
        
        rSquare_x, rSquare_density = getKernelDensity(this_RSquare_list)
        rSquare_x_list.append(rSquare_x)
        rSquare_density_list.append(rSquare_density)
            
    graphTitle = f'Distribution of Estimated β with B = {numberOfSimulations_B}'
    fileName = '1_MC_pdf_beta'
    xName = 'beta'
    distributionPlot(beta_x_list, beta_density_list, time_list, xName, graphTitle, fileName)

    graphTitle = f'Distribution of Estimated t-statistics with B = {numberOfSimulations_B}'
    fileName = '1_MC_pdf_t'
    xName = 't'
    distributionPlot(t_x_list, t_density_list, time_list, xName, graphTitle, fileName)

    graphTitle = f'Distribution of Estimated R^2 with B = {numberOfSimulations_B}'
    fileName = '1_MC_pdf_RSquare'
    xName = 'R^2'
    distributionPlot(rSquare_x_list, rSquare_density_list, time_list, xName, graphTitle, fileName)

# Question 2

def read_csv(filename) -> str:
    output = pd.read_csv(filename, header = 0)
    return output
    
part1Data = read_csv('../3/data_tsde_assignment_3_part_1.csv')
part2Data = read_csv('../3/data_tsde_assignment_3_part_2.csv')

def samplePlot(input_sample, input_x, input_interval, input_ylabel, input_title, input_fileName):
    plot_x = pd.to_datetime(input_x, dayfirst=True, format='%d/%m/%Y')
    
    plt.figure(figsize=(13,5))    
    plt.plot(plot_x, input_sample, linewidth = 1, color = 'blue')
    plt.title(input_title, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel(input_ylabel)    
    
    # Set n-month ticks on the x-axis
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=input_interval))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # Auto-format date labels (avoid overlapping)
    plt.gcf().autofmt_xdate()

    plt.savefig(f'../3/figures/{input_fileName}.jpeg', dpi=300)
    plt.show()

def sacf(input_data, input_lag, input_title, input_fileName):
    result = []
    
    for i in range(1, input_lag+1):
        x = pd.Series(input_data)
        result.append(x.autocorr(lag = i))
    
    lags = np.arange(1,input_lag+1,1)
    
    plt.figure(figsize=(10,6))
    plt.bar(lags, result, color='blue', edgecolor='black')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.ylim(0.8, 1)
    plt.title(input_title, fontweight='bold')
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.savefig(f'../3/figures/{input_fileName}.jpeg', dpi=300)
    plt.show()
    
    return result

def partI_question2():
    aapl_stock = part1Data[['DATE','APPLE']]
    aapl_stock = aapl_stock.set_index('DATE')
    msft_stock = part1Data[['DATE','MICROSOFT']]
    msft_stock = msft_stock.set_index('DATE')

    graphTitle = 'APPLE stock time series'
    fileName = '2_aapl'
    samplePlot(aapl_stock, part1Data['DATE'], 6, 'Daily Stock Price', graphTitle, fileName)

    graphTitle = 'MICROSOFT stock time series'
    fileName = '2_msft'
    samplePlot(msft_stock, part1Data['DATE'], 6, 'Daily Stock Price', graphTitle, fileName)

    input_lags = 12

    graphTitle = 'Sample ACF of AAPL daily stock prices'
    fileName = '2_sacf_aapl'
    aapl_sacf = sacf(aapl_stock['APPLE'], input_lags, graphTitle, fileName)

    graphTitle = 'Sample ACF of MSFT daily stock prices'
    fileName = '2_sacf_msft'
    msft_sacf = sacf(msft_stock['MICROSOFT'], input_lags, graphTitle, fileName)

# Question 3

def generateMatrixX_ADF(input_delta_y, input_y , p):
    numberOfRows = len(input_delta_y) - p
    result_matrix = []
    
    for rowNumber in range(0,numberOfRows):
        row = []
        
        # the first term X_t-1
        row.append(input_y.iloc[p + rowNumber - 1])
        # the following terms of ΔX_t-i
        for t in range(0, p-1):
            row.append(input_delta_y[p + rowNumber - t - 1])
    
        result_matrix.append(row)
     
    return pd.DataFrame(result_matrix)

def generateMatrixX_ECM(input_y, input_x, input_z, p, q):
    numberOfRows = min(len(input_y) - p, len(input_x) - q)
    extra = max(q-p, 0)
    result_matrix = []
    
    for rowNumber in range(0,numberOfRows):
        row = []
        display_Row = []
        row.append(input_z[extra + p + rowNumber - 1])
        display_Row.append(extra + p + rowNumber - 1)
        
        for t in range(0, p):
            row.append(input_y[extra + p + rowNumber - t - 1])
            display_Row.append(extra + p + rowNumber - t - 1)
            
        for t in range(0, q):
            row.append(input_x[extra + p + rowNumber - t - 1])
            display_Row.append(extra + p + rowNumber - t - 1)
    
        result_matrix.append(row)
     
    return pd.DataFrame(result_matrix)

# compute Akaike information criterion
def getAIC(t, k, residuals):
    ssr_k = 0
    
    for res in residuals:
        ssr_k += res ** 2
    
    aic = t * np.log(ssr_k/t) + k * 2
    
    return aic

# compute Bayesian information criterion
def getBIC(t, k, residuals):
    ssr_k = 0
    
    for res in residuals:
        ssr_k += res ** 2
    
    bic = t * np.log(ssr_k/t) + k * np.log(t)
    
    return bic

def getBestP(input_list):
    min_value = round(np.min(input_list),3)
    final_p = input_list.index(np.min(input_list)) + 1
    return min_value, final_p

def adf_test(input_alpha, input_adf, boolean_intercept):
    # DF critical Value without intercept at significance level = 0.1
    criticalValue_withoutIntercept = -1.62
    
    # DF critical Value with intercept at significance level = 0.1
    criticalValue_withIntercept = -2.57
    
    if input_adf < criticalValue_withoutIntercept:
        result = 'Reject the null'
        # print(f'{result}.')
        return result
    else:
        result = 'Do not reject'
        # print(f'{result}.')
        return result

def partI_question3():
    print('\nQuestion 3:')
    stock_list = part1Data.columns.tolist()
    stock_list.pop(0)

    stock_df = pd.DataFrame(columns=['Stock_ID','Order of p', 'ADF', 'Decision'])

    for sNumber, stock in enumerate(stock_list):
        max_p = 12
        hat_beta_list = []
        se_beta_list = []
        hat_residuals_list = []
        bic_list = []
        
        differenceStockPrice = np.diff(part1Data[stock].squeeze())

        # estimate with a maximum p up to 12 lags
        for iterateP in range(1, max_p+1):
            # estimate an AR(p) model with intercept for the given data
            matrix_X, hat_beta, hat_y, hat_residuals = runRegressionModel_ADF(input_delta_y=differenceStockPrice, 
                                                                              input_y=part1Data[stock].squeeze(),
                                                                              input_p=iterateP)
            
            estimate_SE = getStandardError(matrix_X, hat_residuals)
            beta_SE = estimate_SE[0]
            se_beta_list.append(beta_SE)
            
            lengthOfSeries = len(part1Data[stock])
            k = iterateP + 1
            this_bic = getBIC(lengthOfSeries, k, hat_residuals.squeeze())
            
            hat_beta_list.append(hat_beta.iloc[0,0])
            hat_residuals_list.append(hat_residuals)
            bic_list.append(float(this_bic))

        # Use BIC for model selection
        min_bic, final_p = getBestP(bic_list)
        final_beta = hat_beta_list[final_p - 1]
        final_residuals = hat_residuals_list[final_p - 1]
        final_se = se_beta_list[final_p - 1]
        
        # compute the ADF test statistic under the null
        adf_statistic = (final_beta) / final_se
        decision = adf_test(0.1, adf_statistic, False)
        
        stock_df.loc[sNumber] = [stock, final_p, adf_statistic, decision]
    
    stock_df = stock_df.set_index('Stock_ID')  
    print(stock_df)

def cointegrationTestDecision(inputDecision):
    if inputDecision == 'Reject the null':
        return 'Zt ∼I(0): cointegration between {Yt} and {Xt}'
    else:
        return 'Zt ∼I(1): No-cointegration between {Yt} and {Xt}'
    
def performADFCointegrationTest(input_deltaResiduals, input_Zt_list):
    # estimate the ADF regression model
    max_p = 12  # maximum lags
    hat_beta_list = []
    hat_residuals_list = []
    se_beta_list = []
    aic_list = []

    for iterateP in range(1, max_p+1):
        matrix_X, estimate_phi, hat_y, hat_residuals = runRegressionModel_ADF(input_delta_y=input_deltaResiduals, 
                                                                              input_y=input_Zt_list.squeeze(),
                                                                              input_p=iterateP)
        
        estimate_SE = getStandardError(matrix_X, hat_residuals)
        beta_SE = estimate_SE[0]
        se_beta_list.append(beta_SE)

        lengthOfSeries = len(hat_residuals)
        k = iterateP + 1
        hat_beta_list.append(estimate_phi.iloc[0,0])
        hat_residuals_list.append(hat_residuals)
        
        this_aic = getAIC(lengthOfSeries, k, hat_residuals.squeeze())
        aic_list.append(float(this_aic))
    
    min_aic, final_p = getBestP(aic_list)
    final_beta = hat_beta_list[final_p - 1]
    final_residuals = hat_residuals_list[final_p - 1]
    final_se = se_beta_list[final_p - 1]
        
    # compute the ADF test statistic under the null
    adf_statistic = (final_beta) / final_se
    print(f'ADF test statistic: {round(adf_statistic,3)}')

    # perform the ADF-Conintegration test
    decision = adf_test(0.1, adf_statistic, False)
    cointegrated = cointegrationTestDecision(decision)
    print(f'Conclusion regarding cointegration at ⍺ = 10%: {decision} and {cointegrated}')
    
def partI_question4():
    print('\nQuestion 4:')
    
    msft_stock = part1Data[['DATE','MICROSOFT']]
    msft_stock = msft_stock.set_index('DATE')
    xom_stock = part1Data[['DATE','EXXON_MOBIL']]
    xom_stock = xom_stock.set_index('DATE')
    
    # perform a regression of Y_t on X_t
    matrix_X, estimate_beta, hat_y, hat_residuals_z = runRegressionModel(msft_stock.squeeze(), xom_stock.squeeze())
    alpha_hat, beta_hat = estimate_beta.iloc[0,0],  estimate_beta.iloc[1,0]
    
    # compute general test statistics
    estimate_SE = getStandardError(matrix_X, hat_residuals_z)
    t_statistic = estimate_beta.iat[1,0] / estimate_SE[1]
    rSquare = getRSquare(hat_residuals_z.squeeze(), xom_stock.squeeze())

    print('\nRegression Result: MICROSOFT on EXXON_MOBIL ---')
    print(f'α̂ = {estimate_beta.iat[0,0]:.3f}')
    print(f'β̂ = {estimate_beta.iat[1,0]:.3f}')
    print(f't(β) = {t_statistic:.3f}')
    print(f'R² = {rSquare:.3f}')

    # plot the stock prices of MSFT and XOM
    graphTitle = 'Microsoft vs Exxon Mobil Stock Prices'
    fileName = '4_price_msft_xom'
    
    plot_x = pd.to_datetime(msft_stock.index, dayfirst=True, format='%d/%m/%Y')
    
    plt.figure(figsize=(10,5))
    plt.plot(plot_x, msft_stock, label = 'Microsoft', color = 'steelblue')
    plt.plot(plot_x, xom_stock, label = 'Exxon Mobil', color = 'darkorange')
    plt.title(graphTitle, fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel("Price")
    
    # Set n-month ticks on the x-axis
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    # Auto-format date labels (avoid overlapping)
    plt.gcf().autofmt_xdate()
    
    plt.legend()
    plt.savefig(f'../3/figures/{fileName}.jpeg', dpi=300)
    plt.show()

    graphTitle = 'Microsoft vs Exxon Mobil (Level Regression)'
    fileName = '4_scatter_msft_xom'
    
    # plot the scatter plot with OLS regression line
    plt.figure(figsize=(6,6))
    plt.scatter(xom_stock.squeeze(), msft_stock.squeeze(), alpha=0.5, label="Observed")
    plt.plot(xom_stock.squeeze(), hat_y.squeeze(), color='red', label="Fitted line")
    plt.title(graphTitle, fontweight='bold')
    plt.xlabel('Exxon Mobil Stock Price')
    plt.ylabel('Microsoft Stock Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'../3/figures/{fileName}.jpeg', dpi=300)
    plt.show()

    print(f'\nCointegration regression: MSFT = {alpha_hat: .3f} + {beta_hat: .3f} * XOM')
    
    # diﬀerencing of Z_t
    differenceResiduals = np.diff(hat_residuals_z.squeeze())
    
    performADFCointegrationTest(input_deltaResiduals=differenceResiduals, input_Zt_list=hat_residuals_z)
    
partI_question1()
partI_question2()
partI_question3()
partI_question4()

'''
Part II: Cointegration and Error Correction Models
'''
print('\nPart II: Cointegration and Error Correction Models')

# Question 1

def monteCarlo_Cointegrated(input_B, input_t, input_phi, input_corr):
    this_beta_list = []
    this_t_list = []
    this_RSquare_list = []
    this_lambda_list = []
    
    for i in range(0, input_B):
        v_t = np.random.normal(loc=0, scale=1, size=input_t+1)
        w_t = np.random.normal(loc=0, scale=1, size=input_t+1)
    
        this_x_list = []
        this_x_list.append(0)
        this_y_list = []

        for j in range(1, input_t+1):
            x_t = input_phi * this_x_list[j-1] + v_t[j]
            this_x_list.append(x_t)
            y_t = input_corr * this_x_list[j] + w_t[j]
            this_y_list.append(y_t)
            
        this_x_list.pop(0)
        
        matrix_X, estimate_beta, hat_y, hat_residuals = runRegressionModel(this_y_list, this_x_list)
        
        estimate_SE = getStandardError(matrix_X, hat_residuals)
        t_statistic = (estimate_beta.iat[1,0] - input_corr) / estimate_SE[1]
        rSquare = getRSquare(hat_residuals.squeeze(), this_y_list)                

        this_beta_list.append(estimate_beta.iat[1, 0])
        this_t_list.append(t_statistic)
        this_RSquare_list.append(rSquare)
        
        estimate_lambda = estimate_beta.iat[1,0]
        this_lambda_list.append(estimate_lambda)
    
    return this_beta_list, this_t_list, this_RSquare_list, this_lambda_list

def partII_question1():
    time_list = [100, 500, 1000]
    phi_list = [1, 0.5, 0.25]
    numberOfSimulations_B = 5000

    for phi in phi_list:
        beta_x_list = []
        beta_density_list = []
        t_x_list = [] 
        t_density_list = []
        rSquare_x_list = []
        rSquare_density_list = []
        
        lambda_x_list = []
        lambda_density_list = []
        
        for time in time_list:
            corr = 0.5
            
            this_beta_list, this_t_list, this_RSquare_list, this_lambda_list = monteCarlo_Cointegrated(numberOfSimulations_B, time, phi, corr)
                
            beta_x, beta_density = getKernelDensity(this_beta_list)
            beta_x_list.append(beta_x)
            beta_density_list.append(beta_density)
            
            t_x, t_density = getKernelDensity(this_t_list)
            t_x_list.append(t_x)
            t_density_list.append(t_density)
            
            rSquare_x, rSquare_density = getKernelDensity(this_RSquare_list)
            rSquare_x_list.append(rSquare_x)
            rSquare_density_list.append(rSquare_density)
        
            lambda_x, lambda_density = getKernelDensity(this_lambda_list)
            lambda_x_list.append(lambda_x)
            lambda_density_list.append(lambda_density)
        
        if phi == 1:
            graphTitle = f'Distribution of Estimated β with cointegrating vector (1,{-corr})'
            fileName = f'5_MC_pdf_beta_phi_{phi}'
            xName = 'beta'
            distributionPlot(beta_x_list, beta_density_list, time_list, xName, graphTitle, fileName)

            graphTitle = f'Distribution of Estimated t-statistics with cointegration vector (1,{-corr})'
            fileName = f'5_MC_pdf_t_phi_{phi}'
            xName = 't'
            distributionPlot(t_x_list, t_density_list, time_list, xName, graphTitle, fileName)

            graphTitle = f'Distribution of Estimated R^2 with cointegration vector (1,{-corr})'
            fileName = f'5_MC_pdf_RSquare_phi_{phi}'
            xName = 'R^2'
            distributionPlot(rSquare_x_list, rSquare_density_list, time_list, xName, graphTitle, fileName)
        else:
            graphTitle = f'Distribution of Estimated λ with ɸ = {phi}'
            fileName = f'5_MC_pdf_lambda_phi_{phi}'
            xName = 'lambda'
            distributionPlot(lambda_x_list, lambda_density_list, time_list, xName, graphTitle, fileName)

quarterly_name = part2Data['obs']
aggregateConsumption = part2Data['CONS']
aggregateIncome = part2Data['INC']
fourth_quarter_name = [quarter for quarter in quarterly_name if 'Q4' in quarter]

def partII_question2():    
    # plot the graphs
    graphTitle = 'The quarterly aggregate consumption and aggregate income in the Netherlands'
    fileName = '6_cons_inc'

    plt.figure(figsize=(13,5))    
    plt.plot(quarterly_name, aggregateConsumption, 
            linewidth = 1, color = 'green', label='Aggregate consumption')
    plt.plot(quarterly_name, aggregateIncome, 
            linewidth = 1, color = 'blue', label='Aggregate income')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.title(graphTitle, fontweight='bold')
    plt.xticks(quarterly_name, minor=True)
    plt.xticks(fourth_quarter_name, minor=False, rotation=45)
    plt.grid(True, axis='x', which='minor', 
            linestyle='--', linewidth=0.5, color='grey')
    plt.grid(True, axis='x', which='major', 
            linestyle='--', linewidth=0.8, color='black')
    plt.xlabel('Time')
    plt.ylabel('Euros')
    plt.legend()
    plt.savefig(f'../3/figures/{fileName}.jpeg', dpi=300)
    plt.show()
    
    input_lags = 12

    graphTitle = 'Sample ACF of Aggregate Comsumption'
    fileName = '6_sacf_aggCons'
    aggregateConsumption_sacf = sacf(aggregateConsumption, input_lags, graphTitle, fileName)

    graphTitle = 'Sample ACF of Aggregate Income'
    fileName = '6_sacf_aggInc'
    aggregateIncome_sacf = sacf(aggregateIncome, input_lags, graphTitle, fileName)
    
def partII_question3():
    print('\nQuestion 3:')
    # perform a regression of Y_t on X_t
    matrix_X, estimate_beta, hat_y, hat_residuals_z = runRegressionModel(aggregateConsumption, aggregateIncome)
    alpha_hat, beta_hat = estimate_beta.iloc[0,0],  estimate_beta.iloc[1,0]

    print(f'\nCointegration regression: CONSUMPTION = {alpha_hat: .3f} + {beta_hat: .3f} * INCOME')

    # plot the regression residuals
    graphTitle = 'Residuals of ADF regression'
    fileName = '7_res_ADF'
    
    plt.figure(figsize=(13,5))   
    plt.plot(quarterly_name, hat_residuals_z, 
            linewidth = 1, color = 'blue')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.title(graphTitle, fontweight='bold')
    plt.xticks(quarterly_name, minor=True)
    plt.xticks(fourth_quarter_name, minor=False, rotation=45)
    plt.grid(True, axis='x', which='minor', 
            linestyle='--', linewidth=0.5, color='grey')
    plt.grid(True, axis='x', which='major', 
            linestyle='--', linewidth=0.8, color='black')
    plt.xlabel('Time')
    plt.ylabel('Euros')
    plt.savefig(f'../3/figures/{fileName}.jpeg', dpi=300)
    plt.show()
    
    # diﬀerencing of Z_t
    differenceResiduals = np.diff(hat_residuals_z.squeeze())
    
    performADFCointegrationTest(input_deltaResiduals=differenceResiduals, input_Zt_list=hat_residuals_z)

def partII_question4():
    print('\nQuestion 4:')
    # perform a regression of Y_t on X_t
    matrix_X, estimate_beta, hat_y, hat_residuals_z = runRegressionModel(aggregateConsumption, aggregateIncome)
    alpha_hat, beta_hat = estimate_beta.iloc[0,0],  estimate_beta.iloc[1,0]
    
    # diﬀerencing of Y_t and X_t
    differenceConsumption = np.diff(aggregateConsumption.squeeze())
    differenceIncome = np.diff(aggregateIncome.squeeze())
    
    # estimate the Error-Correction Model
    max_p = 6   # maximum lags
    max_q = 6   # maximum lags
    
    hat_coefs_list = []
    hat_residuals_list = []
    spec_list = []
    aic_list = []
    
    for iterateP in range(1, max_p+1):
        for iterateQ in range(1, max_q+1):      
            matrix_X, estimate_coefficient, hat_y, hat_residuals = runRegressionModel_ECM(input_y=differenceConsumption, 
                                                                                          input_x=differenceIncome, 
                                                                                          input_z=hat_residuals_z.squeeze(), 
                                                                                          input_p=iterateP, input_q=iterateQ)
            
            lengthOfSeries = len(hat_residuals)
            k = iterateP + 1
            hat_coefs_list.append(estimate_coefficient.squeeze())
            hat_residuals_list.append(hat_residuals)
            spec_list.append([iterateP,iterateQ])
            
            this_aic = getAIC(lengthOfSeries, k, hat_residuals.squeeze())
            aic_list.append(float(this_aic))
            
    min_aic, final_index = getBestP(aic_list)
    final_spec_p = spec_list[final_index - 1][0]
    final_spec_q = spec_list[final_index - 1][1]
    final_coefs = hat_coefs_list[final_index - 1]
    final_gamma = final_coefs[0]
    final_phis = final_coefs[1:(final_spec_p+1)]
    final_betas = final_coefs[-final_spec_q:]
    
    # report the selected lags of the error-correction model
    print(f'\nSelected lags of ECM by AIC: p = {final_spec_p}, q = {final_spec_q}')

    # report the estimated error-correction model
    print(f'The estimated ECM is: ΔY_t = {round(final_gamma,3)} * Z_t-1 + ', end='')
    
    for index, phi in enumerate(final_phis):
            print(f'{round(phi,3)} * ΔY_t-{index+1} + ', end='')
            
    for index, beta in enumerate(final_betas):
        if index != len(final_betas)-1:
            print(f'{round(beta,3)} * ΔX_t-{index+1} + ', end='')
        else:
            print(f'{round(beta,3)} * ΔX_t-{index+1}')
    
    # report the long-run equilibrium
    print(f'\nLong-run equilibrium of ECM is: CONSUMPTION = {alpha_hat: .3f} + {beta_hat: .3f} * INCOME')

    # interpret the value of the error correction coeﬃcient
    print(f'\nError-Correction coefficient Ɣ = {final_gamma: .3f}')
    if final_gamma < 0:
        print('Interpretation: Negative gamma -> Indicates correction towards long-run equilibrium (speed of adjustment).')
    else:
        print('Interpretation: Non-negative gamma -> No correction to equilibrium (or divergence).')

partII_question1()
partII_question2()
partII_question3()
partII_question4()