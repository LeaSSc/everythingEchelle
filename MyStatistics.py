#%% import
import matplotlib.pyplot as plt
import numpy as np
import os#, sys.math, glob

#from matplotlib import cm
from matplotlib import colors as c
from matplotlib.colors import ListedColormap

from scipy.stats import gaussian_kde#,truncnorm
from scipy.ndimage import gaussian_filter

from scipy.optimize import curve_fit, minimize, basinhopping, differential_evolution

import emcee
import dynesty

try:
    import numdifftools as nd
except:
    print()

from ploting_functions import *

#%% cornerplot
def get_contour_levels(Z, levels=[0.68, 0.95]):
    Zflat = Z.flatten()
    Zsorted = np.sort(Zflat)[::-1]
    cumsum = np.cumsum(Zsorted)
    cumsum /= cumsum[-1]  
    
    levels_new = []
    for l in levels:
        idx = np.searchsorted(cumsum, l)
        if idx >= len(Zsorted):
            idx = len(Zsorted) - 1
        levels_new.append(Zsorted[idx])
    
    levels_new = np.unique(np.sort(levels_new))
    return levels_new

class vonMises_kde:
    def __init__(self, samples, period=2*np.pi, weights=None, kappa=None):
        
        self.period = period
        self.theta = (2*np.pi/period) * (samples % period)
        
        if weights is None:
            weights = np.ones_like(self.theta)
        self.weights = weights / np.sum(weights)
        
        if kappa is None:
            kappa=self.get_kappa()
            # kappa=200
            
        self.kappa=kappa

    def evaluate(self, x):
        from scipy.stats import vonmises
        theta_x = (2*np.pi/self.period) * (x % self.period)

        pdf_theta = np.sum(
            self.weights[:,None] *
            vonmises.pdf(theta_x[None,:], self.kappa, loc=self.theta[:,None]),
            axis=0
        )

        return pdf_theta * (2*np.pi/self.period)
    
    def get_kappa(self):
        R = np.abs(np.sum(self.weights * np.exp(1j * self.theta)))
        
        N_eff = 1.0 / np.sum(self.weights**2)
        sigma_sq = -2 * np.log(R) if R > 0 else 1.0

        bw = (4 / (3 * N_eff))**(1/5) * np.sqrt(sigma_sq)
        return 1.0 / (bw**2)
    
def plot_cornerplot(parameters,labels,xi=None,L=None,labelcbar='',mask=None
                   ,levels=[0.68, 0.95],get_percentiles_using_kde=False
                   ,statistics_type='probability density',bins = 100, median_or_mode='median'
                   ,kde_2D_plot=False,smooth2Dhist=False,idx_for_circular_parameters=[]
                   ,bins_tothist=50,plot=True):
    
    if L is None and not xi is None:
        L=np.exp(-1/2*xi)
        
    if xi is None:
        xi=np.ones_like(parameters[0])
        
        parameters = [p[~np.isnan(xi)] for p in parameters]
        
        xi= xi[~np.isnan(xi)]
        # L=None
    
    for idx_p in idx_for_circular_parameters:
        parameters[idx_p] = parameters[idx_p] % 1
    
    num=len(labels)-1
    xparameters=parameters[:-1]
    xlabels=labels[:-1]
    
    yparameters=parameters[1:]
    ylabels=labels[1:]
    
    match statistics_type:
        case 'weighted probability density' | 'probability density':
            cmap_name='Greys'
            n_colors = 10000
            cmap = plt.colormaps.get_cmap(cmap_name)
            base_colors = cmap(np.linspace(0, 1, n_colors))
        
            
            white = np.array([[1.0, 1.0, 1.0, 0.0]]*50)
            
            newcolors = np.vstack((white,base_colors))
            cmap = ListedColormap(newcolors)
            add1=1
            
            plot_pdf=True
            
        
        case 'raw chi^2' | 'raw likelihood':
            add1=0
            cmap_name='magma'
            cmap = plt.colormaps.get_cmap(cmap_name)
            cmap=c.LinearSegmentedColormap.from_list(cmap,cmap(np.linspace(0, 0.93, 100)))
            
            plot_pdf=False
    

    match statistics_type:
        case 'raw likelihood' | 'weighted probability density':
            weights = L
        case 'raw chi^2':
            weights=xi
        case 'probability density':
            weights=None
    
    
    if plot:
        gridxy=len(parameters)-1
        axes=generate_subplot_grid((gridxy+add1,gridxy+add1),share_x=True,share_y=False,add_cbar=not plot_pdf
                                   ,wspace=0,hspace=0,top=0.90,bottom=0.13,left=0.1,figsize=(10,7))
        if not plot_pdf:
            axes,ax_cbar=axes
            plt.subplots_adjust(bottom=0.1,top=0.9,left=0.10,right=0.90,hspace=0,wspace=0)
                
        for idx_xax in range(num):
            for idx_yax in range(num):
                if not idx_xax>=idx_yax:
                    axes[idx_xax][idx_yax].set_axis_off()
                else:
                    
                    x = xparameters[idx_yax]
                    y = yparameters[idx_xax]
                    
                    x_min, x_max = x.min(), x.max()
                    y_min, y_max = y.min(), y.max()
                    
                    if plot_pdf:
                        idx_xax1=idx_xax+1
                        
                        if kde_2D_plot:
                            w=weights
                            
                            if idx_yax in idx_for_circular_parameters:
                                x_min=0.
                                x_max=1.
                                
                                x_median=np.median(x)
                                x = (x - x_median + 0.5) % 1
                            
                            if idx_xax+1 in idx_for_circular_parameters:
                                y_min=0.
                                y_max=1.
                                
                                y_median=np.median(y)
                                y = (y - y_median + 0.5) % 1
                            
                            xy = np.vstack([x, y])
    
                            kde = gaussian_kde(xy, weights=w)
                            X, Y = np.meshgrid(np.linspace(x_min, x_max, bins),
                                               np.linspace(y_min, y_max, bins))
                            
                            Z = kde(np.vstack([X.ravel(), Y.ravel()]))
                            Z = Z.reshape(X.shape)
                            
                            if idx_yax in idx_for_circular_parameters:
                                X= (X+ x_median - 0.5) %1
                                x= (x+ x_median - 0.5) 
                                
                            if idx_xax+1 in idx_for_circular_parameters:
                                Y= (Y+ y_median - 0.5) %1
                                y= (y+ y_median - 0.5) 
                            
                            dx = (x_max - x_min) / (bins - 1)
                            dy = (y_max - y_min) / (bins - 1)
                            Z /= np.sum(Z) * dx * dy
                            
                            Xcent=X
                            Ycent=Y
                            
                        else:
                            H, xedges, yedges = np.histogram2d(x, y, bins=bins
                                                              ,range=[[x.min(), x.max()], [y.min(), y.max()]]
                                                              ,weights=weights,density=True)
                            xcent = 0.5*(xedges[1:] + xedges[:-1])
                            ycent = 0.5*(yedges[1:] + yedges[:-1])
                            
                            Xcent, Ycent = np.meshgrid(xcent, ycent)
                            X, Y         = np.meshgrid(xedges, yedges)
                            
                            if not smooth2Dhist:
                                Z = H.T  
                            else:
                                if idx_yax in idx_for_circular_parameters and (idx_xax + 1) in idx_for_circular_parameters:
                                    mode = 'wrap'
                                elif idx_yax in idx_for_circular_parameters:
                                    mode = ('wrap', 'nearest') 
                                elif (idx_xax + 1) in idx_for_circular_parameters:
                                    mode = ('nearest', 'wrap')
                                else:
                                    mode = 'nearest'
                                    
                                Z = gaussian_filter(H.T, sigma=2.0, mode=mode)

                                dx = (xedges.max() - xedges.min()) / bins
                                dy = (yedges.max() - yedges.min()) / bins
                                Z /= np.sum(Z) * dx * dy
                        
                        if weights is None:
                            scatter_c='k'
                            vmin=0
                            vmax=1
                        else:
                            scatter_c=np.log10(weights) 
                            vmax=np.sort(scatter_c)[-1]#[-num_models_goodxi]
                            vmin=min(scatter_c)
                        
                        axes[idx_xax1][idx_yax].scatter(x,y,c=scatter_c,cmap='jet',s=0.5,alpha=1,zorder=-30
                                                        # ,vmin=vmin,vmax=vmax 
                                                        )
                        
                        levels_trans = get_contour_levels(Z, levels)
                        
                        Z_min, Z_max = Z.min(), Z.max()
                        # Z_norm = (Z - Z_min) / (Z_max - Z_min)  
                        cut_norm = (levels_trans[0] - Z_min) / (Z_max - Z_min)  
                        
                        n_pre = int(cut_norm * n_colors)
                        base_colors = cmap(np.linspace(0.1, 1, n_colors-n_pre))
    
                        white = np.array([[1.0, 1.0, 1.0, 0.0]] * n_pre)
                        newcolors = np.vstack((white, base_colors))
                            
                        cmapi = ListedColormap(newcolors)
                        
                        p = axes[idx_xax1][idx_yax].pcolormesh(X, Y, Z, cmap=cmapi, shading='auto')#, vmin=0, vmax=1)

                        axes[idx_xax1][idx_yax].contour( Xcent, Ycent, Z, levels=levels_trans, colors='k')
                        axes[idx_xax1][idx_yax].contourf(Xcent, Ycent, Z, levels=[min(levels_trans),Z.max()], colors='white', zorder=-20)
                        
                        avoid_edge_ticks(axes[idx_xax1][idx_yax])
                        axes[idx_xax1][idx_yax].tick_params(axis='x',rotation=30)
    
                    else:
                        order = np.flip(np.argsort(weights))
                        
                        idx_xax1=idx_xax
                        p=axes[idx_xax][idx_yax].scatter(x[order],y[order],c=weights[order],cmap=cmap,s=1
                                                        # ,vmin=min(xi),vmax=np.sort(xi)[num_models_goodxi]
                                                        #,norm=c.LogNorm(vmin=min(xi), vmax=1)
                                                        )
                        if not (mask is None):
                            axes[idx_xax][idx_yax].scatter(x[order][mask[order]],y[order][mask[order]],c='k',s=50
                                                            #,norm=c.LogNorm(vmin=min(xi), vmax=1)
                                                            ,marker='x'
                                                            )
                            
                    if idx_xax==num-1:
                        axes[idx_xax1][idx_yax].set_xlabel(xlabels[idx_yax])
                    else:
                        axes[idx_xax1][idx_yax].tick_params(labelbottom=False,bottom=False)
                    if idx_yax==0:
                        axes[idx_xax1][idx_yax].set_ylabel(ylabels[idx_xax])
                    else:
                        axes[idx_xax1][idx_yax].tick_params(labelleft=False,left=False)
                        
        
        for idx_xax in range(num):
            idx_yax=num
            if not idx_xax>=idx_yax:
                axes[idx_xax][idx_yax].set_axis_off()
    
    uncertainties=[]
    results=[]
    
    for idx_xax in range(num+1):
        idx_yax=idx_xax
        if plot:
            axes[idx_xax][idx_yax].tick_params(labelleft=False,left=False,labelbottom=False,bottom=False,right=True)
        
        samples= parameters[idx_xax]
        if weights is None:
                weights = np.ones_like(samples)
        
        range_parameter=(np.nanmin(samples),np.nanmax(samples))
        hist_weighted,edges     =np.histogram(samples, bins=bins_tothist,range=range_parameter, weights=weights)
       
        hist_sum = np.sum(hist_weighted)
        hist_weighted=hist_weighted/hist_sum
        
        if plot:
            axes[idx_xax][idx_yax].stairs(hist_weighted,edges,lw=1,zorder=10,color='k') 
        
        bw_method=None
        kappa=None
        
        if bins_tothist>100:
            bw_method= 1/bins_tothist*10
            sigma = bw_method * np.std(samples)
            kappa = 1 / sigma**2 /10
            
        if idx_xax in idx_for_circular_parameters:
            circular=True
        else:
            circular=False
        
        bin_width = edges[1] - edges[0]
        hist_area = np.sum(hist_weighted) * bin_width 
        
        result,uncertainty,kde,median=evaluate_sampler_result(samples,weights,circular=circular,bw_method=bw_method,kappa=kappa
                                ,median_or_mode=median_or_mode,get_percentiles_using_kde=get_percentiles_using_kde)
        
        uncertainties.append(uncertainty)
        results.append(result)
        
        if plot:
            x = np.linspace(samples.min(), samples.max(), 10000)
            y_kde=kde(x)
            y_scaled = y_kde * hist_area
            axes[idx_xax][idx_yax].plot(x, y_scaled , color='r')
            
            axes[idx_xax][idx_yax].axvline(uncertainty[0],color='lightgrey',ls=':')
            axes[idx_xax][idx_yax].axvline(uncertainty[1],color='lightgrey',ls=':')
            axes[idx_xax][idx_yax].axvline(result   ,color='red'      ,ls=':')
            axes[idx_xax][idx_yax].axvline(median     ,color='k'        ,ls='--')
            
            
            axes[idx_xax][idx_yax].set_title(rf"{labels[idx_xax]} = $ {result:.3f}^{{+{uncertainty[1]:.3f}}}_{{-{uncertainty[0]:.3f}}} $")
            
            avoid_edge_ticks(axes[idx_xax1][idx_yax])
            # avoid_edge_ticks(axes[idx_xax][idx_xax])
            
    if plot:
        axes[0][0].tick_params(labelleft=False,left=True,labelbottom=False,bottom=False,right=True)
        # axes[0][0].set_ylabel('P')
        axes[num][num].tick_params(labelleft=False,left=False,labelbottom=True,bottom=True,right=True)
        axes[num][num].set_xlabel(labels[-1])
        axes[num][num].tick_params(axis='x',rotation=30)

                
        if not plot_pdf:
            plt.colorbar(p,extend='max',cax=ax_cbar)
            
            ax_cbar.set_title(r'$χ^2_{{{}}}$'.format(labelcbar))
            
    return np.array(results),uncertainties
            
def evaluate_sampler_result(samples,weights,circular=False,median_or_mode='mode',get_percentiles_using_kde=False
                            ,bw_method=None,kappa=None):
    x = np.linspace(samples.min(), samples.max(), 10000)
    if circular:
        kde = vonMises_kde(samples, weights=weights,period=1.,kappa=kappa)
    else:
        kde = gaussian_kde(samples, weights=weights,bw_method=bw_method)
        
    y = kde.evaluate(x)
        
    mode_idx = np.argmax(y)
    mode_kde = x[mode_idx]
    
    if circular:
        x = (x - mode_kde + 0.5) % 1
        idx_sort = np.argsort(x)
        x = x[idx_sort]
        y = y[idx_sort]
        
        samples= (samples-mode_kde+0.5)%1
    
    idx_sort = np.argsort(samples)
    samples_sorted = samples[idx_sort]
    weights_sorted = weights[idx_sort]
    weights_normalized = weights_sorted/np.sum(weights)
    
    if not median_or_mode=='mode':
        if get_percentiles_using_kde:
            y_norm=y/np.sum(y)
            cdf = np.cumsum(y_norm)#*(x[1]-x[0]) #!!! *(x[1]-x[0])
            x_temp=x
        else:
            cdf = np.cumsum(weights_normalized) - 0.5 * weights_normalized
            x_temp=samples_sorted
            
        lower_bound,median,upper_bound  = np.interp([0.16,0.5,0.84], cdf, x_temp)
            
    else: #Chen-Shao algorithm 
        cdf = np.cumsum(weights_normalized)
        
        left=np.arange(len(samples))
        right_to_each_left = np.searchsorted(cdf, cdf + 0.68) #cdf[j] - cdf[i] >= 0.68
        
        mask  = right_to_each_left < len(samples)
        left  = left[mask]
        right = right_to_each_left[mask]
        widths = samples_sorted[right] - samples_sorted[left]
        
        min_idx = np.argmin(widths)
        
        lower_bound = samples_sorted[left [min_idx]]
        upper_bound = samples_sorted[right[min_idx]]
        
        cdf_temp = cdf - 0.5 * weights_normalized
        median = np.interp(0.50, cdf_temp, samples_sorted)
    
    if median_or_mode=='mode':
        result=mode_kde
    elif median_or_mode=='median':
        result=median
    # elif median_or_mode=='mean':
    #     result=np.average(samples,weights=weights)

    if circular:
        if median_or_mode == 'mode':
            center_shifted = 0.5
        elif median_or_mode == 'median':
            center_shifted = median
            
        lower_err = center_shifted - lower_bound
        upper_err = upper_bound - center_shifted
        
        lower_bound = (lower_bound + mode_kde - 0.5) % 1
        upper_bound = (upper_bound + mode_kde - 0.5) % 1
        median      = (median      + mode_kde - 0.5) % 1
    
    if median_or_mode=='mode':
        result=mode_kde
    elif median_or_mode=='median':
        result=median
    
    if not circular:
        lower_err = result - lower_bound
        upper_err = upper_bound - result

    uncertainty=[lower_err, upper_err]
    
    return result,uncertainty,kde,median

#%% fitters
def neg_log_likelihood_X22dof(params=None ,x=None, y=None, func=False, model=False, y_fit=False):
    if isinstance(y_fit,bool):
        if not isinstance(func,bool):
            y_fit = func(x, *params)
        elif not isinstance(model,bool):
            if params is not None:
                model.parameters = params
            y_fit = model(x)
            
    y_fit = np.clip(y_fit, 1e-20, None)
    return np.sum(np.log(y_fit) + y / y_fit)
        
class Chi2TwoDofFitter:
    def __init__(self, minimizer='minimize',MCMC=False,labels=['fwhm','amp','x0']):
        self.minimizer=minimizer
        self.labels=labels
        self.MCMC=MCMC
        
    def __call__(self, model, x, y):
        p0 = model.parameters
        bounds = [model.bounds[name] for name in model.param_names]
        
        def _to_minimize(params):
            if self.MCMC:
                modeli = model.copy()
            else:
                modeli=model
            modeli.parameters = params
            return neg_log_likelihood_X22dof(params=params, x=x, y=y, model=modeli)
        
        if self.MCMC:
            def log_prob(params):
                for p, (low, high) in zip(params, bounds):
                    if not (low < p < high):
                        return -np.inf
                return -_to_minimize(params)
            
            ndim = len(p0)
            nwalkers = 2 * ndim + 2
            nsteps = 2000
            bins=30

            # pos = p0 + 1e-4 * np.random.randn(nwalkers, ndim)
            pos = np.array([np.random.uniform(low=b[0], high=b[1], size=nwalkers) for b in bounds]).T

            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
            sampler.run_mcmc(pos, nsteps, progress=True)

            self.sampler = sampler
            self.samples = sampler.get_chain(discard=500, flat=True)#,thin=10)
            weights=None
            
            statistics_type='probability density'
            results,uncertainties=plot_cornerplot(self.samples.T,labels=self.labels,xi=None,L=weights,labelcbar='',mask=None, levels=[0.68, 0.95]
                           ,statistics_type=statistics_type,bins = bins,median_or_mode='mode'
                           ,kde_2D_plot=False,plot=True)
            
            model.parameters = results
            return model,uncertainties
            
        else:
            if self.minimizer=='minimize':
                result = minimize(_to_minimize,p0,method='L-BFGS-B',bounds=bounds)
            elif self.minimizer=='differential_evolution':
                result=differential_evolution(_to_minimize,bounds=bounds)
                if not result.success:
                    print('Chi2TwoDofFitter failed')

            model.parameters = result.x
        return model
    
class JointChi2TwoDofFitter:
    def __init__(self, joint_parameter_idx=[],joint_parameter_p0=[],joint_parameter_bounds=[]
                     ,joint_parameter_function=False,joint_para_always_positive=False,model_x0_index=0
                     ,minimizer='minimize'):
        if len(joint_parameter_idx)>1 and not isinstance(joint_parameter_function,bool):
            print(" wont work! ToDo")
        
        self.joint_parameter_idx    = joint_parameter_idx
        self.joint_parameter_p0     = joint_parameter_p0
        self.joint_parameter_bounds = joint_parameter_bounds
        
        self.joint_parameter_function=joint_parameter_function
        self.model_x0_index=model_x0_index
        
        self.minimizer=minimizer
        self.joint_para_always_positive=joint_para_always_positive
        
    def __call__(self, models, xs, ys):
        
        joint_parameter_idx    = self.joint_parameter_idx
        joint_parameter_p0     = self.joint_parameter_p0
        joint_parameter_bounds = self.joint_parameter_bounds
        
        p0=[]
        bounds=[]
        
        params_per_model=len(models[0].parameters)-len(joint_parameter_idx)
        
        for modeli in models: 
            p0.extend(np.delete(modeli.parameters,joint_parameter_idx))
            boundi = np.array([modeli.bounds[name] for name in modeli.param_names])
            bounds.extend(np.delete(boundi,joint_parameter_idx,axis=0))
            
        p0.extend(joint_parameter_p0)
        bounds.extend(joint_parameter_bounds)
        
        def _to_minimize(parameters):
            joint_params  = parameters[-len(joint_parameter_p0):]
            params= parameters[:-len(joint_parameter_p0)]
            
            total_logL = 0.0
            for i_model,(modeli,xi,yi) in enumerate(zip(models,xs,ys)):
                paramsi=params[i_model*params_per_model:(i_model+1)*params_per_model]
                
                for i_jparams_idx in range(len(joint_parameter_idx)): 
                    jparams_idx=joint_parameter_idx[i_jparams_idx]
                    
                    if isinstance(self.joint_parameter_function,bool):
                        jparams    =joint_params[i_jparams_idx]
                    else:
                        x0 = paramsi[ self.model_x0_index ] 
                        jparams= self.joint_parameter_function(x0,*joint_params)
                        
                        if self.joint_para_always_positive and np.any(jparams <= 0):
                            return 1e20
                        
                    paramsi=np.insert(paramsi,jparams_idx,jparams)
                
                modeli.parameters = paramsi

                total_logL += neg_log_likelihood_X22dof(params=paramsi,x=xi,y=yi,model=modeli)

            return total_logL

        if self.minimizer=='minimize':
            result = minimize(_to_minimize,p0,method='L-BFGS-B',bounds=bounds)
        elif self.minimizer=='differential_evolution':
            result=differential_evolution(_to_minimize,bounds=bounds)
            if not result.success:
                print('JointChi2TwoDofFitter failed')
        
        jresult=result.x[-len(joint_parameter_p0):]
        self.joint_parameter_result=jresult
        for i_model in range(len(models)):
            resulti=result.x[i_model*params_per_model:(i_model+1)*params_per_model]
            
            for i_jparams_idx in range(len(joint_parameter_idx)): 
                jparams_idx=joint_parameter_idx[i_jparams_idx]
                
                if isinstance(self.joint_parameter_function, bool):
                    jresulti= jresult[i_jparams_idx]
                else:
                    x0 = resulti[self.model_x0_index]
                    jresulti = self.joint_parameter_function(x0, *jresult)
                
                resulti=np.insert(resulti,jparams_idx,jresulti)
            
            models[i_model].parameters=resulti
        return models
    
class Fitter():
    def __init__(self
                     ,fitting_method='minimize' #options: minimize,differential_evolution,basinhopping,MCMC,dynesty
                     ,likelihood='chi2'         #options: chi2,chi22dof
                     
                     ,minimizer_options = {"method": "L-BFGS-B"
                                          ,"options": {"ftol": 1e-14, "gtol": 1e-12, "maxiter": 5000}}
                     ,scale_free_parameters=False
                     
                     ,polish=True,inflate_bounds = 1.0
                     
                     #MCMC
                     ,nwalkers_factor=10,nsteps=2000,thin=False,burnin=200
                     ,set_walkers_percentage_around_initial_guess=False #if False equaly distribute inside boarders
                     
                     #dynesty
                     ,nlive=None
                     
                     #minimizers
                     ,niterations_for_fit=False
                     ,popsize=None
                     
                     #cornerplot
                     ,get_percentiles_using_kde=False,bins = 100,median_or_mode='median'
                     ,kde_2D_plot=False,smooth2Dhist=False,idx_for_circular_parameters=[]
                     ,bins_tothist=50
                     
                     #
                     ,x=None,y=None,y_sig=1,bounds=None,func=None
                     
                     ):
        
        self.minimizer=fitting_method
        self.likelihood=likelihood
        
        self.minimizer_options=minimizer_options
        self.scale_free_parameters=scale_free_parameters
        
        self.polish=polish
        self.inflate_bounds=inflate_bounds
        
        #MCMC
        self.nwalkers_factor=nwalkers_factor
        self.nsteps=nsteps
        self.thin=thin
        self.burnin=burnin
        self.set_walkers_percentage_around_initial_guess=set_walkers_percentage_around_initial_guess
        
        #dynesty
        self.nlive=nlive
        
        #minimizers
        self.niterations_for_fit=niterations_for_fit
        self.popsize=popsize
        
        #cornerplot
        self.median_or_mode=median_or_mode
        self.get_percentiles_using_kde=get_percentiles_using_kde
        self.bins=bins
        self.kde_2D_plot=kde_2D_plot
        self.smooth2Dhist=smooth2Dhist
        self.idx_for_circular_parameters=idx_for_circular_parameters
        self.bins_tothist=bins_tothist
        
        self.func=func
        self.bounds=bounds
        self.x=x
        self.y=y
        self.y_sig=y_sig
        
        self.input_kwargs = {}
        self._calc_hessian_with_bounds = False
        
        if not bounds is None:
            self.mins = np.array([b[0] for b in self.bounds])
            self.maxs = np.array([b[1] for b in self.bounds])
        
    def set_parameters_to_fit(self,labels=None,**kwargs):
        
        self.parameter_names=list(kwargs.keys())
        parameter_names_flat=[]
        parameter_structure=[]
        initial_guess=[]
        for k, v in kwargs.items():
            if isinstance(v,(list,np.ndarray)):
                for i in range(len(v)):
                    parameter_names_flat.append(k+f'_{i}')
                    initial_guess.append(v[i])
                parameter_structure.append(len(v))
            else:
                parameter_names_flat.append(k)
                initial_guess.append(v)
                parameter_structure.append(1)
                
        self.initial_guess_flat=initial_guess
        self._structure=parameter_structure
        
        if labels is None:
            self.labels=parameter_names_flat
        else:
            self.labels=labels
        
    def _reshape(self, flat):
        unflat = []
        idx = 0
        for length in self._structure:
            if length == 1:
                unflat.append(flat[idx])
            else:
                unflat.append(flat[idx:idx+length])
            idx += length
        return unflat
                    
    def set_additional_inputs(self,**kwargs):
        self.input_kwargs=kwargs
        

    def _normalize(self,p0):
        return (p0 - self.mins) / (self.maxs - self.mins)
    def _denormalize(self,params):
        if self.scale_free_parameters:
            return self.mins + params * (self.maxs - self.mins)
        else:
            return params

        
    def _to_minimize(self,y_fit):
        if self.likelihood=='chi2':
            return np.sum(((self.y-y_fit)/self.y_sig)**2)
            
        elif self.likelihood=='chi22dof':
            y_fit = np.clip(y_fit, 1e-20, None)
            return np.sum(np.log(y_fit) + self.y / y_fit)
            
    def _to_optimise(self,params):
        if self.scale_free_parameters:
            params=self._denormalize(params)
            
        parameters=self._reshape(params)
        kwargs = dict(zip(self.parameter_names, parameters))
        
        if self.minimizer=='MCMC' or self._calc_hessian_with_bounds:
            if np.any(params < self.mins) or np.any(params > self.maxs):
                if self._calc_hessian_with_bounds:
                    return 1e20
                return -np.inf
        
        y_fit=self.func(self.x,kwargs,self.input_kwargs)
    
        if self.minimizer in ['MCMC','dynesty']:
            return -self._to_minimize(y_fit=y_fit)
        else:
            return self._to_minimize(y_fit=y_fit)
        
    def fit(self,func=None,x=None,y=None,y_sig=None,bounds=None,plot=True):
        
        if not y is None:
            self.x=x
            self.y=y
        if not y_sig is None:
            self.y_sig=y_sig
        if not bounds is None:
            self.bounds=bounds
            
            self.mins = np.array([b[0] for b in self.bounds])
            self.maxs = np.array([b[1] for b in self.bounds])
        if not func is None:
            self.func=func
        
        if self.scale_free_parameters:
            initial_guess=self._normalize(self.initial_guess_flat)
            bounds=[(0., 1.)] * len(initial_guess)
        else:
            initial_guess=self.initial_guess_flat
            bounds=self.bounds

        ndim = len(initial_guess)
        weights = None
        ######################## MCMC ##############################
        if self.minimizer in ['MCMC','dynesty']:
            if self.minimizer=='MCMC':
                nwalkers = ndim*self.nwalkers_factor
                if self.set_walkers_percentage_around_initial_guess:
                    if isinstance(self.set_walkers_percentage_around_initial_guess,bool):
                        f=1
                    else:
                        f=self.set_walkers_percentage_around_initial_guess
                    pos = initial_guess + f*np.random.randn(nwalkers, ndim)
                else:
                    pos = np.array([np.random.uniform(low=b[0], high=b[1], size=nwalkers) for b in bounds]).T
        
                sampler = emcee.EnsembleSampler(nwalkers, ndim, self._to_optimise)
                sampler.run_mcmc(pos, self.nsteps, progress=True)
                self.raw_result  = sampler
                self.samples = sampler.get_chain(discard=self.burnin,flat=True,thin=self.thin) 
                
                self.statistics_type='probability density'
        ######################## dynesty ##############################
            elif self.minimizer=='dynesty':
                def prior_transform(u):
                    if self.scale_free_parameters:
                        return u
                    v = np.zeros(ndim)
                    for i in range(ndim):
                        low, high = bounds[i]
                        v[i] = low + u[i] * (high - low)
                    return v
                
                sampler = dynesty.DynamicNestedSampler(self._to_optimise, prior_transform, ndim=ndim,nlive = self.nlive)
                sampler.run_nested()#dlogz=0.1) #!!!
            
                res = sampler.results
                
                self.raw_result = sampler
                self.samples  = res.samples
                self.weights = np.exp(res.logwt - res.logz[-1])
                
                weights=self.weights
            
                self.statistics_type='weighted probability density'
                
            results,uncertainties=plot_cornerplot(self.samples.T,labels=self.labels,xi=None,L=weights
                        ,levels=[0.68, 0.95],get_percentiles_using_kde=False
                        ,statistics_type=self.statistics_type,bins = self.bins, median_or_mode=self.median_or_mode
                        ,kde_2D_plot=self.kde_2D_plot,smooth2Dhist=self.smooth2Dhist,idx_for_circular_parameters=self.idx_for_circular_parameters
                        ,bins_tothist=self.bins_tothist,plot=plot)
            
            self.result=results
            self.uncertainties=uncertainties
            
            ############################# polish ############################
            if self.polish:
                bounds_tight = [(result - self.inflate_bounds * sig[0],result + self.inflate_bounds * sig[1]) for result, sig in zip(results, uncertainties)]
                
                minimizer_temp=self.minimizer
                self.minimizer='minimize'
                result_minimize=minimize(self._to_optimise,results,method=self.minimizer_options['method'],bounds= bounds_tight,options=self.minimizer_options['options'])
                self.minimizer=minimizer_temp
                
                self.result_polished=result_minimize.x
        
        else:
            
            if self.niterations_for_fit:
                niterations_for_fit=self.niterations_for_fit+1
            else:
                niterations_for_fit=1
                
            self.results         =[0]*niterations_for_fit
            if self.polish:
                self.results_polished=[0]*niterations_for_fit
            
            for n_iter in range( niterations_for_fit):
                if self.minimizer=='minimize':
                    result_n=minimize(self._to_optimise,initial_guess,method=self.minimizer_options['method'],bounds=bounds,options=self.minimizer_options['options'])
                    self.results[n_iter]=self._denormalize(result_n.x)
                    
                elif self.minimizer=='differential_evolution':
                    print("start ",n_iter)
                    result_n=differential_evolution(self._to_optimise,bounds=bounds,polish=False,tol=1e-2,popsize=self.popsize)
                    self.results[n_iter]= self._denormalize(result_n.x) 
                    print('polish')
                    result_n=minimize(self._to_optimise,result_n.x,method=self.minimizer_options['method'],bounds=bounds
                                      ,options=self.minimizer_options['options'])
                                      
                    self.results_polished[n_iter]= self._denormalize(result_n.x) 
                    print("finished ",n_iter)
        
                elif self.minimizer == 'basinhopping':
                    result_n = basinhopping(self._to_optimise, initial_guess,  minimizer_kwargs=self.minimizer_options
                                            ,niter=10000)
                    self.results[n_iter]= self._denormalize(result_n.x) 
                    
                if n_iter==0:
                    cov = result_n.hess_inv.todense()
                    self.uncertainties = np.sqrt(np.diag(cov))
                    print('hessian')
                    try:
                        self._calc_hessian_with_bounds=True
                        H = nd.Hessian(self._to_optimise)(result_n.x)
                        
                        cov = np.linalg.inv(H)
                        self.uncertainties = np.sqrt(np.diag(cov)) 
                    except:
                        print('num Hessian failed')
                        
            self.result=self.results[0]
                        
        return self._reshape(self.result)
        
    
