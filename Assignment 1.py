import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from datetime import datetime, timedelta
##################
#Data extarction
#Bonds_data
data = np.loadtxt('bonds.txt', delimiter=",", dtype=str) #xtract data
BONDS = [] #holds the bonds information
for line in data:
    bond = {}
    prices = line[3:13]
    prices = [float(price) for price in prices]
    bond["prices"] = prices
    bond["coupon"] = float(line[0])
    bond["issue"] =  datetime.strptime(line[2], "%m/%d/%Y")
    bond["maturity"] =  datetime.strptime(line[1], "%m/%d/%Y")
    BONDS.append(bond)
#days data was taken
date_strings = ["1/6/2025", "1/7/2025", "1/8/2025", "1/9/2025", "1/10/2025",
                "1/13/2025", "1/14/2025", "1/15/2025", "1/16/2025", "1/17/2025"]
days_data = [datetime.strptime(date, "%m/%d/%Y") for date in date_strings]

################
#Helper functions

def get_paynment_times(price_date, issued, maturity):
    """
    Calculates the dates at which payments must occur for a given bond

    Parameters:
    price_date (datetime): date at which the price was measured
    issued (datetime): issuing date of the bond
    issued (datetime): maturity date of the bond

    Returns:
    days_payment (List[int]): the days at which coupon payments must occur, as a number of days from price date
    coupon_at_maturity (boolean): if there is a coupon at maturity with its full value (true), or its prorated (false)
    days_since_last_payment (int): number of days from last coupon payment until price_date (for dirty price)
    days_prorated_coupon (int): if the coupon at maturity its discounted, number of days since last full coupon payment to maturity

    """
    times_for_payment = []
    coupon_at_maturity = False

    date_counter = issued
    last_payment = issued #tracks last payment before price date
    payment_before_maturity = issued #used to see if there is a smaller last coupon payment
    while date_counter <= maturity:
        if date_counter > price_date:
            times_for_payment.append(date_counter)
        next_month = date_counter.month + 6
        next_year = date_counter.year
        if next_month > 12:
            next_month -= 12
            next_year += 1
        if date_counter < price_date:
            last_payment = date_counter
        payment_before_maturity = date_counter
        date_counter = date_counter.replace(year=next_year, month=next_month)
    if maturity in times_for_payment:
        coupon_at_maturity = True #indicates that the last coupon payment occurs with the nominal
    else:
        times_for_payment.append(maturity)

    days_payment = [(day - price_date).days for day in times_for_payment] #transforms the dates to number of days from price_date

    days_since_last_payment = (price_date - last_payment).days #days since the last coupon payment to the date when price was measured

    days_prorated_coupon = 0
    if not coupon_at_maturity:
        days_prorated_coupon = (maturity - payment_before_maturity).days #days from the last scheduled coupon payment until maturity

    return days_payment, coupon_at_maturity, days_since_last_payment, days_prorated_coupon


def present_value(ytm, bond, days, coupon_at_maturity, days_prorated_coupon):
    """
    Calculates the present value at which payments must occur for a given bond
    for ytm calculation.

    Parameters:
    ytm (float): ytm
    bond (dict): the bond we want to calculate
    days (List[int]): dates at which payments occur
    coupon_at_maturity [boolean]: if there is a coupon at maturity
    days_prorated_coupon [int]: number of days between last full coupon and maturity

    Returns:
    float: present value of the bond assuming ytm
    """
    if coupon_at_maturity: #if last coupon is paid at maturity
        return np.sum(bond["coupon"]/2*np.e**(-ytm*days/365)) + 100*np.e**(-ytm*days[-1]/365)
    else: #if an irregular coupon is paid at maturity
        return np.sum(bond["coupon"]/2*np.e**(-ytm*days[:len(days)-1]/365)) + 100*np.e**(-ytm*days[-1]/365) + bond["coupon"]*days_prorated_coupon/365*np.e**(-ytm*days[-1]/365)


def dirty_price(data_num, bond, days_since_last):
    """
    Calculates the dirty price for bond

    Parameters:
    data_num (int): index to identify the clean price
    bond (dict): the bond we want to calculate
    days_since_last [int]: number of days since the last coupon payment

    Returns:
    float: dirty price of the bond
    """
    price = bond["prices"][data_num]
    return price + days_since_last/365*bond["coupon"]



def get_r(spot_rates, spot_rates_times, day, today):
    """
    Calculates the rate at a given time through interpolation

    Parameters:
    spot_rates (List[float]): list with the already calculated spot rates
    spot_rates_times (List[datetime]): list with the dates of the spot_rates
    today (datetime): date at which we are analyzing the spot rate data
    day [int]: number of days since today at which we want to obtain the spot rate

    Returns:
    float: interpolated spot rate at time today+day
    """
    added_days = timedelta(days=day)
    time = today + added_days
    if time in spot_rates_times: #if we have the spot rate for that time
        return spot_rates[np.where(spot_rates_times == time)[0][0]]
    else:
        if time > np.max(spot_rates_times): #if the time we are searching is larger than any recorded
            return spot_rates[-1] #choose the last spot rate
        index = 0
        while spot_rates_times[index] < time:
            index += 1
        #linear interpolation between the spot rates
        r = spot_rates[index-1] + (spot_rates[index]-spot_rates[index-1]) * (time-spot_rates_times[index-1]).days/(spot_rates_times[index]-spot_rates_times[index-1]).days
        return r

def get_fwrd_rates(fw_rates, fw_times):
    """
    Calculates the forward rates at 1_1yr, 1_2yr, 1_3yr, 1_4yr through
    interpolation of the fw rates at different times.

    Parameters:
    fw_rates (List[float]): list with the already calculated forward rates
    fw_times (List[datetime]): list with the dates of the forward rates
    Returns:
    list[float]: rates for 1_1yr, 1_2yr, 1_3yr, 1_4yr
    """
    index = 3
    fw1_1yr = fw_rates[index-1] + (fw_rates[index]-fw_rates[index-1]) * (datetime(2027, 1, 6)-fw_times[index-1]).days/(fw_times[index]-fw_times[index-1]).days
    index = 5
    fw1_2yr = fw_rates[index-1] + (fw_rates[index]-fw_rates[index-1]) * (datetime(2028, 1, 6)-fw_times[index-1]).days/(fw_times[index]-fw_times[index-1]).days
    index = 7
    fw1_3yr = fw_rates[index-1] + (fw_rates[index]-fw_rates[index-1]) * (datetime(2029, 1, 6)-fw_times[index-1]).days/(fw_times[index]-fw_times[index-1]).days
    return [fw1_1yr, fw1_2yr, fw1_3yr, fw_rates[-1]]

###############
#ytm, spot rate and forward rate calculations

def get_ytm(bonds):
    """
    Calculates and plots the ytm curve. YTM is obtained for each bond and each
    day data was taken. fsolve solves for present_value = dirty_price to get the
    ytm for a given bond and day. Assumes instantaneous compounding.

    Parameters:
    bonds [dict]: contains all the information about the bonds

    Returns:
    ytm_all (List[List[float]]): list with the calculated ytm for each day of data
    ytm_times_all (List[List[datetime]): list with the dates of the ytm for each day of data
    """
    ytm_all = []
    ytm_times_all = []
    plt.figure()
    for i, today in enumerate(days_data): #iterating over every day were data was taken

        yield_day_maturities=[datetime(2025, 1, 6)] #used to store the ytm maturities
        yield_day_ytm=[0] #used to store the ytm

        for bond in bonds:
            days, coupon_at_maturity, days_since_last, pro_rate_days = get_paynment_times(today, bond["issue"], bond["maturity"])
            ytm_solution = fsolve(lambda ytm: present_value(ytm, bond, days, coupon_at_maturity, pro_rate_days) - dirty_price(i, bond, days_since_last), 0.035) #0.035 is the initial guess
            yield_day_maturities.append(bond["maturity"])
            yield_day_ytm.append(float(ytm_solution)*100)

        #data saving and plotting
        ytm_all.append(yield_day_ytm[1:])
        ytm_times_all.append(yield_day_maturities[1:])
        yield_day_maturities.append(datetime(2030, 1, 6))
        yield_day_ytm.append(yield_day_ytm[-1]) #interpolate last ytm as the previous calculated
        yield_day_ytm[0] = yield_day_ytm[1] #add the 0 year as the first ytm calculated
        plt.plot(yield_day_maturities,yield_day_ytm, label=today)

    plt.xlabel("Date")
    plt.ylabel("ytm (%)")
    plt.title("ytm from 6th January, 5 years forward")
    plt.legend()
    plt.savefig("ytm.png")
    plt.grid()
    plt.show()

    return ytm_all, ytm_times_all


def bootstrap_yield_curve(bonds):
    """
    Calculates and plots the spot rate curve. spot rate is obtained for each
    maturity date and each day data was taken. The spot rate is calculated with
    bootstrapping using previously calculated spot rates. Assumes instantaneous
    compounding.

    Parameters:
    bonds [dict]: contains all the information about the bonds

    Returns:
    spot_rates_all (List[List[float]]): list with the calculated spot rates for each day of data
    spot_rates_times_all (List[List[datetime]): list with the dates of the spot rates for each day of data
    """
    spot_rates_all = [] #stores the spot rates
    spot_rates_times_all = [] #stores the times at which spot rates are calculated
    plt.figure()

    for j, today in enumerate(days_data): #iterating over every day were data was taken

        spot_rates = [0]
        spot_rates_times = [datetime(2025, 1, 6)]

        for i, bond in enumerate(bonds): #getting the bonds at one price ordered by maturity
            coupon_rate = bond["coupon"]
            days, coupon_at_maturity, days_since_last, pro_rate_days = get_paynment_times(today, bond["issue"], bond["maturity"])
            price = dirty_price(j, bond, days_since_last)
            for day in days[:len(days)-1]: #discount cashflows (except maturity cashflow)
                price -= coupon_rate/2*np.e**(-get_r(spot_rates,spot_rates_times,day,today)*day/365)
            #spot rate calculation for r(maturity)
            if coupon_at_maturity:
                spot_rates.append((np.log(coupon_rate/2 + 100) - np.log(price))/days[-1] * 365)
                spot_rates_times.append(bond["maturity"])
            else:
                spot_rates.append((np.log(100 + coupon_rate*pro_rate_days/365) - np.log(price))/days[-1] * 365)
                spot_rates_times.append(bond["maturity"])
            if i == 0:
                spot_rates[0] = spot_rates[1]

        #data saving and plotting
        spot_rates_all.append(spot_rates[1:])
        spot_rates_times_all.append(spot_rates_times[1:])
        spot_rates[-1] = spot_rates[-2]
        spot_rates_times[-1] = datetime(2030, 1, 6)
        spot_rates_percentage = [rate * 100 for rate in spot_rates]
        plt.plot(spot_rates_times[2:],spot_rates_percentage[2:], label=today) #only bonds 1year forward are plotted
    plt.xlabel("Date")
    plt.ylabel("ytm (%)")
    plt.title("Spot rate starting 6th January calculation")
    plt.legend()
    plt.savefig("Yield_curve.png")
    plt.grid()
    plt.show()

    return spot_rates_all, spot_rates_times_all


def get_forward_rates(spot_rates_all, spot_rates_times_all):
    """
    Calculates and plots the forward rate curve starting 6th jannuary 2026.
    The forward rate is calculated using the spot rates. Assumes instantaneous
    compounding.

    Parameters:
    spot_rates_all (List[List[float]]): list with the calculated spot rates for each day of data
    spot_rates_times_all (List[List[datetime]): list with the dates of the spot rates for each day of data

    Returns:
    forward_rates_all (List[List[float]]): list with the calculated forward rates for each day of data
    forward_rates_times_all (List[List[datetime]): list with the dates of the forward rates for each day of data
    """
    forward_rates_all = []
    forward_rates_times_all = []
    for j, today in enumerate(days_data): #for each day of data
        spot_rates = spot_rates_all[j]
        spot_rates_times = spot_rates_times_all[j]
        index = 2
        #1year spot rate using interpolation
        spot_rate_1_year = spot_rates[index-1] + (spot_rates[index]-spot_rates[index-1]) * (datetime(2026, 1, 6)-spot_rates_times[index-1]).days/(spot_rates_times[index]-spot_rates_times[index-1]).days
        forward_rate=[0]
        forward_rate_times=[datetime(2026, 1, 6)]
        #using the spot rates to obtain forward rates
        for i in range(index, len(spot_rates)):
            forward_rate.append((spot_rates[i]*(spot_rates_times[i]-datetime(2025, 1, 6)).days - spot_rate_1_year*366)/(spot_rates_times[i]-datetime(2026, 1, 6)).days)
            forward_rate_times.append(spot_rates_times[i])

        #data saving and plotting
        forward_rate[0]=forward_rate[1]
        forward_rate.append(forward_rate[-1])
        forward_rate_times.append(datetime(2030, 1, 6))
        forward_rate_percentage = [rate * 100 for rate in forward_rate]
        plt.plot(forward_rate_times,forward_rate_percentage, label=today)
        forward_rates_all.append(forward_rate.copy())
        forward_rates_times_all.append(forward_rate_times.copy())

    plt.xlabel("Date")
    plt.ylabel("Forward rate (%)")
    plt.title("Forward from 6th January 2026, 4 years forward")
    plt.legend()
    plt.savefig("Forward_rate.png")
    plt.grid()
    plt.show()
    return forward_rates_all, forward_rates_times_all



########
#Main code
ytm_all, ytm_times_all = get_ytm(BONDS) #YTM calculation and plotting
spot_rates_all, spot_rates_times_all = bootstrap_yield_curve(BONDS) #spot rate calculation, only bonds with maturity of more than 1 year from now are plotted
forward_rates_all, forward_rates_times_all = get_forward_rates(spot_rates_all, spot_rates_times_all) #returns the forward rates

#Covariance matrices
yield_returns = np.copy(ytm_all)
x_i_j = np.zeros((5, len(days_data)-1))
index = 0
for i in range(3,8): #five maturities picked arbitrarely (other selection gives similar results)
    for j in range(len(days_data)-1):
        x_i_j[index,j]=np.log(yield_returns[j+1][i]/yield_returns[j][i])
    index += 1
cov_log_yields = np.cov(x_i_j, rowvar=True) #covariance between yields


yield_returns_spot_rate = np.copy(spot_rates_all)
f_i_j = np.zeros((4,len(days_data)-1))
forward_rates = []
for day in range(len(days_data)): #get 1-1yr, 1-2yr- 1-3yr, 1-4yr for each day of data
    forward_rates.append(get_fwrd_rates(forward_rates_all[day], forward_rates_times_all[day]))
for i in range(4):
    for j in range(len(days_data)-1):
        f_i_j[i][j] = np.log(forward_rates[j+1][i]/forward_rates[j][i])
cov_log_fw = np.cov(f_i_j, rowvar=True) #covariance between yields
gir
#Eigenvalue and eigenvectors
print("\nCovariance for yields daily log-returns \n")locals()
print(cov_log_yields)
eigenvalue_yield, eigenvector_yield = np.linalg.eig(cov_log_yields)
print("\neigenvectors")
print(eigenvector_yield)
print("\neigenvalues")
print(eigenvalue_yield)
print("\nCovariance for forward rates daily log-returns \n")
print(cov_log_fw)
eigenvalue_fw, eigenvector_fw = np.linalg.eig(cov_log_fw)
print("\neigenvectors")
print(eigenvector_fw)
print("\neigenvalues")
print(eigenvalue_fw)

