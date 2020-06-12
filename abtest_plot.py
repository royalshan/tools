import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from scipy.stats import norm


# calculate confidence interval
def cal_ci(mean,alpha,sd):
    
    scale = abs(stats.norm.ppf((1-alpha)/2))
    
    lower_bnd = mean - scale*sd
    upper_bnd = mean + scale*sd
    
    return (lower_bnd, upper_bnd)

def plot_dist(cr_a, cr_b, n_a, n_b, xrange):
    
    std_a = np.sqrt(cr_a * (1-cr_a)/n_a)
    std_b = np.sqrt(cr_b * (1-cr_b)/n_b)
    
    prob_a = norm(cr_a, std_a).pdf(xrange)
    prob_b = norm(cr_b, std_b).pdf(xrange)

    # Make the bar plots
    fig, ax = plt.subplots(figsize=(15,5))

    ax.plot(xrange, prob_a, label="Control")
    ax.axvline(x=cr_a, c='blue',alpha=0.5, linestyle='--' )
    ax.plot(xrange, prob_b, label="Test")
    ax.axvline(x=cr_b, c='red',alpha=0.5, linestyle='--' )
    
    ax.legend(frameon=False)
    plt.xlabel("Conversion rate"); 
    plt.ylabel("Probability");
    plt.title("Experiment Test")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1))

def plot_lift(lift, se):
    
    p = norm(lift,se)
   
    # plot conversion rate lift
    x = np.linspace(lift*0.88, lift*1.12, 1000)
    y = p.pdf(x)
    
    # area_under_curve = p.sf(0)
    fig, ax = plt.subplots(figsize=(15,5))
    plt.plot(x, y, label="PDF")
    ax.axvline(x=lift, c='blue',alpha=0.5, linestyle='--' )
    
    # plt.fill_between(x, 0, y, where=x>0, label="Prob(b>a)", alpha=0.3)
    # plt.annotate(f"change={lift:.1%}", (lift, 100))
    plt.title("Difference in conversion rate")
    plt.xlabel("Difference in conversion rate"); 
    plt.ylabel("Probability");

    xi = [lift -3*se, lift -2*se, lift-se, lift, lift + se, lift +2*se, lift +3*se]
    xi_label = [ f"{v:.2%}" for v in xi]
    plt.xticks(xi, xi_label)