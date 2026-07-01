#%% import
import matplotlib.pyplot as plt
import numpy as np
import os#, sys.math, glob
try:
    import pandas as pd
except:
    print()  

# import jax.numpy as jnp

import numbers

from astropy.io import fits 
#from astropy.time import Time,TimeDelta
from astropy.table import Table, Row, vstack

#import astropy.units as u
#from astropy.units import cds

#from matplotlib import cm
from matplotlib import colors as c

from astropy.modeling import models as astropy_models
from astropy.modeling import fitting
from astropy.modeling.models import custom_model

from astropy.convolution import Gaussian1DKernel,Box1DKernel, convolve
from astropy.timeseries import LombScargle
from astropy.stats import sigma_clip

from scipy import stats
from scipy.signal import savgol_filter#, cwt, ricker
from scipy.signal import find_peaks 
from scipy.optimize import curve_fit, minimize, basinhopping, differential_evolution
from scipy.interpolate import interp1d#, pchip_interpolate
from scipy.stats import truncnorm

#import pygyre as pg
#import mesa_reader as mr

try:
    import lightkurve as lk
except:
    print('no lightkurve')

from matplotlib.widgets import TextBox, Button

# from uncertainties import ufloat, unumpy
# from uncertainties.umath import sqrt, log10 

import emcee
# import corner
import dynesty
# from dynesty import utils as dyfunc

try:
    import numdifftools as nd
except:
    print()

import time

import sys

from MyStatistics import *
from ploting_functions import *
#plt.close('all')
# plot_format(color='k')


#%% variables
class Variables():
    def __init__(self,s=None,variable_names=None,x=False,data=None,_scalar_flags=None):
        self.s = s
        self.variable_names = variable_names or []
        
        if data is not None :
            self._data = data
            self._scalar_flags = [np.isscalar(col) for col in self._data]
            self._structure = [len(col) if isinstance(col, (list, np.ndarray)) else 1 for col in self._data]
    
        elif s is not None:
            if not isinstance(x, bool):
                if len(x) != len(variable_names):
                    raise ValueError("Length of x must match length of variable_names")
                else:
                    for name, vals in zip(variable_names, x):
                        setattr(self.s, name, vals)
            self.refresh_data()
        if len(self.variable_names)>0:
            self.basename=variable_names[0].split("_")[0][:-1]
        else:
            self.basename = "noName"
    
    def refresh_data(self):
            self._data = []
            self._scalar_flags = []
            for name in self.variable_names:
                val = getattr(self.s, name)
                
                if np.isscalar(val):
                    col, flag = val, True
                elif isinstance(val,(np.ndarray,Table)):
                    col, flag = val, False
                else:
                    col, flag = list(val), False
                if self.s is not None:
                    setattr(self.s, name, col)
                    
                self._data.append(col)
                self._scalar_flags.append(flag)
            self._structure = [len(col) if isinstance(col, (list, np.ndarray)) else 1 for col in self._data]
                  
    def __getitem__(self, i):
        
        if isinstance(i, (int, np.integer)):
            if self.s is not None:
                self.refresh_data()
            val = self._data[i]
            return val
            
        elif isinstance(i, str):
            if self.s is not None:
                new_names = []
                for idx, name in enumerate(self.variable_names):
                    col = getattr(self.s, name)[i]
                    
                    new_name = f"{self.basename}{idx}_{i}"
                    setattr(self.s, new_name, col)
                    new_names.append(new_name)
                return Variables(s=self.s, variable_names=new_names)
            
            else:
                new_data = []
                for col in self._data:
                    if isinstance(col, dict):
                        new_data.append(col[i])
                    elif hasattr(col, "__getitem__") and not np.isscalar(col):
                        new_data.append(col[i])
                        
                return Variables(data=new_data)

    def __setitem__(self, i, value):
        if self.s is not None:
            setattr(self.s, self.variable_names[i], value)
        self._data[i] = value
        
    def __len__(self):
        return len(self._data)
    
    def __repr__(self):
        if self.s is not None and self.variable_names:
            variables = [getattr(self.s, name) for name in self.variable_names]
            return f"Variables({dict(zip(self.variable_names, variables))})"
        else:
            return f"Variables(data={self._data})"
                    
    def _apply_op(self, other, op):
        if self.s is not None:
            self.refresh_data()

        def iter_col(col):
            if isinstance(col, (list, np.ndarray)):
                return col
            else:
                return [col]
            
        return_as_array= all(isinstance(datai, np.ndarray) for datai in self._data)
        
        if other is None:
            new_data = [[op(a, None) for a in iter_col(col)] for col in self._data]
        if isinstance(other, Variables):
            new_data = [[op(a, b) for a, b in zip(iter_col(col_a), iter_col(col_b))]
                        for col_a, col_b in zip(self._data, other._data)]
        else:
            new_data = [[op(a, other) for a in iter_col(col)] for col in self._data]
        
        if return_as_array:
            new_data = [np.array(col) for col in new_data]
        return Variables(data=new_data)
    
    def __neg__(self): return self._apply_op(None, lambda x, _: -x)
    def __pos__(self): return self._apply_op(None, lambda x, _: +x)

    def __add__(self, other):      return self._apply_op(other, lambda x, y: x + y)
    def __radd__(self, other):     return self._apply_op(other, lambda x, y: y + x)
    def __sub__(self, other):      return self._apply_op(other, lambda x, y: x - y)
    def __rsub__(self, other):     return self._apply_op(other, lambda x, y: y - x)
    def __mul__(self, other):      return self._apply_op(other, lambda x, y: x * y)
    def __rmul__(self, other):     return self._apply_op(other, lambda x, y: y * x)
    def __truediv__(self, other):  return self._apply_op(other, lambda x, y: x / y)
    def __rtruediv__(self, other): return self._apply_op(other, lambda x, y: y / x)
    def __pow__(self, other):      return self._apply_op(other, lambda x, y: x ** y)
    def __rpow__(self, other):     return self._apply_op(other, lambda x, y: y ** x)
    
    def __eq__(self, other):  return self._apply_op(other, lambda x, y: x == y)
    def __ne__(self, other):  return self._apply_op(other, lambda x, y: x != y)
    def __lt__(self, other):  return self._apply_op(other, lambda x, y: x <  y)
    def __le__(self, other):  return self._apply_op(other, lambda x, y: x <= y)
    def __gt__(self, other):  return self._apply_op(other, lambda x, y: x >  y)
    def __ge__(self, other):  return self._apply_op(other, lambda x, y: x >= y)
    
    def __and__(self, other): return self._apply_op(other, lambda x, y: x and y)
    def __or__(self, other):  return self._apply_op(other, lambda x, y: x or y)
    def __invert__(self):     
        if self.s is not None:
            self.refresh_data()
        return Variables(data=[[not x for x in col] for col in self._data])
    
    def flatten(self):
        flat = []
        for col in self._data:
            if isinstance(col, (list, np.ndarray)):
                flat.extend(col)
            else:
                flat.append(col)
        return flat

    def _reshape(self, flat):
        new_data = []
        idx = 0
        for length in self._structure:
            if length == 1:
                new_data.append(flat[idx])
            else:
                new_data.append(flat[idx:idx+length])
            idx += length
        return new_data
    
    def _prepare_flat_inputs(self, args):
    
        num_elements = sum(self._structure)
        flat_inputs = []
    
        for argi in args:
            if isinstance(argi, Variables):
                vals = argi.flatten()
            elif np.isscalar(argi):
                vals = [argi] * num_elements
            else:
                vals = list(argi)
    
            if len(vals) != num_elements:
                # Reject non-broadcastable arguments early
                raise ValueError(
                    f"Argument has length {len(vals)}, expected {num_elements}"
                )
    
            flat_inputs.append(vals)
    
        return flat_inputs

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        if self.s is not None:
            self.refresh_data()
    
        flat_inputs = self._prepare_flat_inputs(args)
        flat_result = [ufunc(*inputi, **kwargs) for inputi in zip(*flat_inputs)]
        new_data = self._reshape(flat_result)
        
        if all(isinstance(datai, np.ndarray) for datai in self._data):
            new_data = [np.array(col) for col in new_data]
            
        return Variables(data=new_data)
    
    
    def __array_function__(self, func, types, args, kwargs):
        if not any(issubclass(t, Variables) for t in types):
            return NotImplemented
    
        if self.s is not None:
            self.refresh_data()
            
        if func.__name__ in ['ones_like', 'zeros_like', 'empty_like']:
            new_data = [func(col, **kwargs) for col in self._data]
            return Variables(data=new_data)
        
        if func is np.interp:
            x, xp, fp = args
        
            if isinstance(x, Variables):
                x_cols = x.get_data()
                out = [np.interp(np.asarray(col), xp, fp) for col in x_cols]
                return Variables(data=out)
            else:
                result = np.interp(np.asarray(x), xp, fp)
                return Variables(data=[result])
    
        try:
            flat_inputs = self._prepare_flat_inputs(args)
            flat_result = [func(*inputi, **kwargs) for inputi in zip(*flat_inputs)]
        except Exception:
            return NotImplemented
    
        new_data = self._reshape(flat_result)
        return Variables(data=new_data)

    
    def get_data(self):
        return self._data

def make_func_accept_Variables(func):
    def wrapper(x, *args, **kwargs):
        if isinstance(x, Variables):
            out_data = []
            for col in x.get_data(): 
                arr = np.asarray(col)
                out_data.append(func(arr, *args, **kwargs))
            return Variables(data=out_data)
        elif isinstance(x, (list, tuple)) and any(isinstance(el, (list, tuple, np.ndarray)) for el in x):
            return [wrapper(el, *args, **kwargs) for el in x]
        else:
            return func(x, *args, **kwargs)
    return wrapper
    

#%% basics
    #%%%
def remove_nan(x,y=False,return_mask=False,unmask=False):
    if unmask:
        try:
            x=x.unmasked
        except:
            x=x
        try:
            y=y.unmasked
        except:
            y=y
    
    mask=~np.isnan(x)
    x = x[mask]
    
    if not isinstance(y, bool):
        y=y[mask]
        if return_mask:
            return x,y,mask
        return x,y
    
    if return_mask:
        return x,mask
    return x

def my_sort(x,y,return_mask=False):
    
    i_sort = np.argsort(x)
    x=np.array(x)[i_sort]
    if not isinstance(y[0],(list,np.ndarray)):
        y=[y]
    
    for i in range(len(y)):
        y[i]=np.array(y[i])[i_sort]
        
    return x, *y

def smoothing(freq,psd,method='gauss',plot=True,bins=5001,
              wlength=101,polyorder=2,stddev=40,boxw=80,do_bin=False,log_bins=False): #'box','savgol'
    
    if method== 'gauss' or plot == True:
        g = Gaussian1DKernel(stddev=stddev)
        gauss = convolve(psd, g)
    
    if method== 'box' or plot == True:
        b = Box1DKernel(boxw)
        box = convolve(psd, b)
    
    if method== 'savgol' or plot == True:
        savgol = savgol_filter(psd,window_length=wlength,polyorder=polyorder) #Savitzky-Golay filter
    
    if plot:
        labels=['unsmoothed','box','gauss','savgol']
        m_idx=labels.index(method)
        methodes = [psd,box,gauss,savgol]
           
        ax=[0]*len(labels)
        
        plt.figure(num='smoothing',figsize=(16,9))
        for i,smoothi in enumerate(methodes):
        
            bin_meansi, bin_edges, binnumber = stats.binned_statistic(freq,smoothi, 
                statistic='mean', bins=np.logspace(np.log10(min(freq)),np.log10(max(freq)),bins))
            
            bin_freqi = (bin_edges[1:]-bin_edges[:-1])/2+bin_edges[:-1]
            
            ax[i]=plt.subplot(411+i,zorder=-i)
                       
            plt.plot(freq, psd, '-',color='grey')
            plt.plot(freq, smoothi, 'k-',label=labels[i])
            plt.plot(bin_freqi,bin_meansi,'+r',ms=1,label='binned')
            plt.xscale('log')
            plt.yscale('log')
            
            plt.legend()
            plt.ylabel('PSD (ppm² μHz$^{-1}$)')
            
            if not(i == len(labels)-1):
                ax[i].tick_params(labelbottom=False)
                
            ax[i].sharex(ax[0])
            ax[i].sharey(ax[0])
            
            if i == m_idx:
                bin_freq = bin_freqi
                bin_means = bin_meansi
        
        
        plt.xlabel('ν (μHz)')
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0)
        
    else:
        if method == 'gauss':
            smoothi = gauss
        elif method == 'box':
            smoothi = box
        elif method == 'savgol':
            smoothi = savgol
            
        if not do_bin:
            return freq,smoothi
        
        if log_bins:
            bins=np.logspace(np.log10(min(freq)),np.log10(max(freq)),bins)
        else:
            bins=np.linspace(min(freq),max(freq),bins)
        bin_means, bin_edges, binnumber = stats.binned_statistic(freq,smoothi, 
            statistic='mean', bins=bins)
        
        bin_freq = (bin_edges[1:]-bin_edges[:-1])/2+bin_edges[:-1]
       
    nanmask = ~np.isnan(bin_means)
    
    return bin_freq[nanmask],bin_means[nanmask]

def table2dict(table):
    if isinstance(table, Table):
        return {name: np.array(table[name]) for name in table.colnames}
    elif isinstance(table, Row):
        return {name: np.array([table[name]]) for name in table.colnames}
    elif isinstance(table, dict):
        return {name: table2dict(var) for name, var in table.items()}
    elif isinstance(table, (list, tuple)):
        return [table2dict(tablei) for tablei in table]
    return table

def dict2table(dictio):
    if isinstance(dictio, dict) and all(isinstance(v, (list, np.ndarray)) for v in dictio.values()):
        return Table(dictio)
    
    elif isinstance(dictio, list):
        tables = [dict2table(element) for element in dictio]
        if len(tables) == 1 and isinstance(tables[0], (Table,list)):
            return tables[0]
        return tables
    else:
        return dictio
    
def outer_tolist(array_or_arrays):
    if not isinstance(array_or_arrays, np.ndarray) or array_or_arrays.dtype != object:
        return array_or_arrays
    return [outer_tolist(arrayi) for arrayi in array_or_arrays]

def _nearest(tomatch,tosearch ):
    if len(np.array(tosearch).shape) == 0:
        return tosearch * (0 * tomatch + 1)
    if len(np.array(tomatch).shape) == 0:
        return tosearch[np.argmin(np.abs(tosearch - tomatch))]
    return tosearch[np.argmin(np.abs(tomatch[:, None] - tosearch[None, :]), axis=1)]

nearest=make_func_accept_Variables(_nearest)

    #%%% basic func
def slope(x,k,d):
    return k*x+d

def lorentzian(x, x0, amp, fwhm):
    return amp * ((fwhm / 2) ** 2) / ((x - x0) ** 2 + (fwhm / 2) ** 2)
    # return amp / (1.0 + 4.0*((x-x0)/fwhm)**2)
    
def lorentz_sum(x, x0, amp, fwhm):
    dx = x[:,None] - x0
    return np.sum(amp*((fwhm/2)**2)/(dx**2+(fwhm/2)**2), axis=1)

def circular_mean(phases,weights=None,period=1): #[0,1)
    
    radian = phases*(2*np.pi/period)
    
    Cos = np.average(np.cos(radian),weights=weights)
    Sin = np.average(np.sin(radian),weights=weights)
    mean    = np.arctan2(Sin, Cos)   * (period/(2*np.pi))
    radius  = np.sqrt(Cos**2+Sin**2)
    stddev  = np.sqrt(-2*np.log(radius)) * (period/(2*np.pi))
    
    if mean < 0:
        mean += period
        
    return mean, stddev 

#%% get data
def get_fits_data(filename,get_header=False,print_header=False,dataindex=1):
    hdu = fits.open(filename)
    data = hdu[1].data
    header  = hdu[0].header  
    if print_header:
        print(repr(header))
        #print(hdu.info)
    hdu.close() 
    if get_header:
            return header,data
    else: return data
    

def read_table(filename,astropyTable=True,printInstruction=False):
    #return_keys: .colnames
    # .to_pandas
    fileformat= filename.split('.')[-1]
    if fileformat=='fits' or fileformat=='fit':
        if printInstruction:
            print("[header],data=get_fits_data(filename,get_header,print_header,dataindex)")
        data=get_fits_data(filename)
        if astropyTable:
            data=Table(data)
        
    elif fileformat=='csv':
        if astropyTable:
            data=Table.read(filename,format="csv")
            if printInstruction:
                print("data=Table.read(filename,format='csv')")
        else:
            data = pd.read_csv(filename)
            if printInstruction:
                print("data = pd.read_csv(filename) [.to_records]")
                print("(https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)")
                
    elif fileformat=='dat':
        if astropyTable:
            data=Table.read(filename,format="ascii")
            if printInstruction:
                print("data=Table.read(filename,format='ascii')")
        else:
            if printInstruction:
                print("ToDo")
                
    elif fileformat=="vot":
        data = Table.read(filename, format='votable')
        if printInstruction:
            print("data=Table.read(filename,format='vot')")
        else:
            if printInstruction:
                print("ToDo")
                
    return data
#%% LC & PSD
    #%%% LC
def download_kepseismic(starname, cadence="80d-filter", directory="kepseismic_files"):
    from urllib.request import urlretrieve
    os.makedirs(directory, exist_ok=True)
    kicID =starname[4:]
    kic_str = str(kicID).zfill(9)
    kic_prefix = kic_str[:4] + "00000"
    kic_suffix = kic_str[-5:]
    
    url = (f"https://archive.stsci.edu/hlsps/kepseismic/{kic_prefix}/{kic_suffix}/{cadence}/"
           f"hlsp_kepseismic_kepler_phot_kplr{kic_str}-{cadence.split('-')[0]}_kepler_v1_cor-psd-filt-inp.fits")
         
    #lightcurve
    # url = (f"https://archive.stsci.edu/hlsps/kepseismic/{kic_prefix}/{kic_suffix}/{cadence}/"
    #        f"hlsp_kepseismic_kepler_phot_kplr{kic_str}-{cadence.split('-')[0]}_kepler_v1_cor-filt-inp.fits")

    filename = directory+ f"/KIC_{kic_str}.fits"

    urlretrieve(url, filename)
    print(f"Downloaded {filename}")
    return filename
    
def get_Mast_lc(starname='KIC 010841730',author='Kepler',cadence=None,key='flux',normalize=False,to_ppm=False
               ,plot=True,plot_all=False,axlc=False,color="k",marker='x',label='',ylabel=False,ms=2
               ,sigmaclip=999,minCut=[],maxCut=[]):
    
    if not plot:
        plot_all=False
    if not ylabel:
        if normalize and not to_ppm:
            ylabel='Normalized ' +key
        elif to_ppm:
            normalize=True
            ylabel='Flux [ppm]'
        else:
            ylabel='Flux'
    
    if author=='KEPSEISMIC':
        file=download_kepseismic(starname)
        data=read_table(file)
        
        return data['TIME'].value*1e6, data['FLUX'].value, 2450000 #!!!
    
    #author='TESS-SPOC'  SPOC
    elif author=='TESS':
        MAST_search_result_20 = lk.search_lightcurve(starname, author='SPOC',exptime=20)
        MAST_search_result_120 = lk.search_lightcurve(starname, author='SPOC',exptime=120)
        MAST_search_result_tess = lk.search_lightcurve(starname, author='TESS-SPOC')
        
        if len(MAST_search_result_120)>0:
            if not len(MAST_search_result_20)==len(MAST_search_result_120):
                mask_both120_20=[]
                for i_sec,sector in enumerate(MAST_search_result_120.mission):
                    if sector in MAST_search_result_20.mission:
                        mask_both120_20.append(False)
                    else:
                        mask_both120_20.append(True)
                
        mask_onlyTESS=[] 
        for sector in MAST_search_result_tess.mission:
            if not sector in MAST_search_result_120.mission:
                mask_onlyTESS.append(True)
            else:
                mask_onlyTESS.append(False)
        
        MAST_search_result_120=MAST_search_result_120[mask_both120_20]
        MAST_search_result_tess=MAST_search_result_tess[mask_onlyTESS]
        
        lc_collection=[]
        for MAST_i in [MAST_search_result_20,MAST_search_result_120,MAST_search_result_tess]:
            if len(MAST_i)>0:
                print('download: ',MAST_i)
                lc_collection_i=MAST_i.download_all()
                if len(lc_collection)==0:
                    lc_collection=lc_collection_i
                else:
                    for lightcurve in lc_collection_i:
                        lc_collection.append(lightcurve)
    else:     
        MAST_search_result = lk.search_lightcurve(starname, author=author, cadence=cadence)

        lc_collection = MAST_search_result.download_all()
    
    if author=='Kepler':
        JD=lc_collection[0].meta['BJDREFI']
    elif author=='TESS':
        JD=0
    else:
        print('ToDo: get JD')
        return
    
    if plot_all:
        plt.figure()
    
    flux_all = []
    time_all = []
    time_minmax_sections=np.zeros((len(lc_collection),2))
    for idx_lc in range(len(lc_collection)):
        
        flux=lc_collection[idx_lc][key].value
        time=lc_collection[idx_lc]['time'].value
        
        time_minmax_sections[idx_lc]=[np.min(lc_collection[idx_lc]['time'].value),np.max(lc_collection[idx_lc]['time'].value)]
        
        for idx_cut in range(len(minCut)):
            mask=(time>minCut[idx_cut]) & (time<maxCut[idx_cut])
            time=time[~mask]
            flux=flux[~mask]
        
        if normalize:
            flux = flux/np.median(flux)
            
            if to_ppm:
                flux =(flux -1)*1e6
                
        mask_sigmaclip=np.abs(flux-1)<sigmaclip*np.std(flux)
        
        if plot_all:
            plt.plot(time,flux)
            plt.plot(time[~mask_sigmaclip],flux[~mask_sigmaclip],'or')
            
        time=time[mask_sigmaclip]
        flux=flux[mask_sigmaclip]

        flux_all = np.hstack((flux_all,flux))
        time_all = np.hstack((time_all,time))
    
    flux,time=remove_nan(flux_all,time_all,unmask=True)
    
    time,flux=my_sort(time,flux)
    time_minmax_sections = time_minmax_sections[np.argsort(time_minmax_sections[:, 0])]
    
    if plot:
        plot_lightcurve(time=time,flux=flux,JD=JD,axlc=axlc,color=color,marker=marker,label=label,ylabel=ylabel,ms=ms)
    
    return time, flux, JD, time_minmax_sections
            
def plot_lightcurve(time=False,flux=False,axlc=False,JD=2400000
                   ,color="k",marker='x',label='',ylabel="Flux",xlabel=False
                   ,ls='None',ms=2,zorder=None
                    ):
    
    if not xlabel:
        xlabel=f"Time - {JD} [BKJD]"

    if not axlc:
        plt.figure()
        axlc=plt.gca()
        
    axlc.plot(time,flux,marker=marker,ms=ms,color=color,label=label,zorder=zorder,ls=ls)
    
    axlc.set_xlabel(xlabel)
    axlc.set_ylabel(ylabel)


    #%%% PSD
        #%%%% plot
def plot_PSD(freq=False,psd=False,axPSD=False,nu_max=False,zoom=False,flip=False,
             logaxis=False,psdColor='k',color_modes=['limegreen','dodgerblue',(255/256,171/256,53/256),'orange'],color_nu_max_line='grey',modes=False,label=None
            ,alpha=1,**kwargs):
    
    if modes and not isinstance(modes[0],(list,np.ndarray)):
        modes=[modes]
    
    if not axPSD:
        plt.figure()
        axPSD   = plt.gca()

    ylabel= 'PSD (ppm²/μHz)'
    
    if not flip:
        axPSD.plot(freq,psd,color=psdColor,linewidth=1,label=label,alpha=alpha,**kwargs)
        axPSD.set_xlabel('ν (μHz)')
        axPSD.set_ylabel(ylabel)
    else:
        axPSD.plot(psd,freq,color=psdColor,linewidth=1,label=label,alpha=alpha,**kwargs)
        axPSD.set_ylabel('ν (μHz)')
        axPSD.set_xlabel(ylabel)
    
    if logaxis:
        axPSD.set_xscale('log')
        axPSD.set_yscale('log')
    # else:
    #     axPSD.set_xscale('linear')
    #     axPSD.set_yscale('linear')
    
    if nu_max:
        
        if not isinstance(nu_max, (list, np.ndarray)):
            nu_max=np.array([nu_max])
            
        for nu_m in nu_max:
            if not flip:
                axPSD.axvline(nu_m,color=color_nu_max_line,alpha=0.5)
            else:
                axPSD.axhline(nu_m,color=color_nu_max_line,alpha=0.5)
              
        if zoom:
            zoom = nu_m*zoom
            mask = (freq>(nu_m-zoom)) & (freq<(nu_m+zoom))
            maxPSD = max(psd[mask])
            
            if modes:
                
                linestyleFreqSp = ['_','_','_','_']
                markersizeFreqSp = [30,20,10,5]
                markeredgewidth  = [3,1.5,3,1]
                #markeredgewidthBig  = [2,2,2,2]
                
                if len(color_modes)<=len(modes):
                    color_modes.insert(3, 'orange')
                    
                for l in range(len(modes)):
                    if isinstance(color_modes[l],list):
                        color_modes[l]=color_modes[l][0]
                    axPSD.plot(np.ones_like(modes[l])*maxPSD,modes[l],linestyleFreqSp[l],
                            color=color_modes[l],markersize=markersizeFreqSp[l],mew=markeredgewidth[l])
            
            if not flip:
                axPSD.set_xlim(nu_m-zoom,nu_m+zoom)
                axPSD.set_ylim(0-maxPSD*0.075,maxPSD+maxPSD*0.075)
            else:
                axPSD.set_ylim(nu_m-zoom,nu_m+zoom)
                axPSD.set_xlim(0-maxPSD*0.075,maxPSD+maxPSD*0.075)
                axPSD.invert_xaxis()
                
    return axPSD

def apply_highpassfilter(time, flux, width_microhertz=5, sigma_flag=False, sigma_width=3.5, hpf_flag=True):
    time,flux=time,flux=remove_nan(time,flux)
    
    if hpf_flag:
        cadence = np.median(np.diff(time))
        width_cycdays = 1/(60*60*24)*1e6/(width_microhertz)
        filter_width = width_cycdays/cadence
        highlysmoothed_flux = convolve(flux, Gaussian1DKernel(filter_width),boundary='extend')
        hpfiltered_flux = flux/highlysmoothed_flux
    else:
        hpfiltered_flux=flux
    
    if sigma_flag:
        clipped_flux = sigma_clip(hpfiltered_flux, sigma=sigma_width, maxiters=1)
        time = time[~clipped_flux.mask]
        hpfiltered_flux = clipped_flux.data[~clipped_flux.mask]
        
    return time, hpfiltered_flux

        #%%%% calc 
def get_nyqist_freq(t,t_in_sec=False,return_in_muHz=True): 
    if not t_in_sec:
        t=t*(24*60*60)
    
    nyq_freq = (1/(2*(np.median(np.diff(t)))))
    if abs(nyq_freq-283.212e-6)<1e-6:
        nyq_freq=283.212e-6
    else:
        idx_gaps=np.where(np.diff(t)>(24*60*60))[0]
        idx_gaps=[0]+list(idx_gaps)+[-1]
        for i_idx_gap in range(1,len(idx_gaps)):
            t_i=t[idx_gaps[i_idx_gap-1]:idx_gaps[i_idx_gap]]
            nyq_freq = np.min([nyq_freq,(1/(2*(np.median(np.diff(t_i)))))])
            
    if return_in_muHz:
        nyq_freq=nyq_freq*1e6
    return nyq_freq

def calc_lomb_scargle(t,y,oversample=1,amplitude=False,method="fast",nyq_freq=False,max_freq=10000,time_in_sec=False,return_in_ppm=True):
    
    if not time_in_sec:
        t=t*(24*60*60)
        
    t,y = remove_nan(t,y)
    
    df = 1.0 / (t[-1] - t[0])
    fmin = df  
    
    if isinstance(nyq_freq, bool):
        nyq_freq=get_nyqist_freq(t,t_in_sec=True,return_in_muHz=False)
        
    max_freq=min([nyq_freq,max_freq*1e-6])
    
    freq = np.arange(fmin, max_freq, df / oversample)
    model = LombScargle(t, y)
    
    sc = model.power(freq, method=method, normalization="psd")
    
    amp = 2*np.sqrt(sc/len(t))  
    
    if return_in_ppm:
        amp*=1e6
        
    if not amplitude:
        amp=amp**2 * 1/(df*1e6) #df to muhz
    
    return freq*1e6, amp

def calc_psd(time,flux,hpf_flag=False,sigma_clip=False,width_microhertz=5,oversample=1,JD=2454833
             ,plot=True,axpsd=False,logaxis=True
             ,plot_LC=False,return_lc=False,plot_nu_max=False,to_ppm=True,label=None
             ):
    #dont use ppm lc!!
    
    if sigma_clip:
        sigma_flag=True
    else:
        sigma_flag=False

    time,flux=apply_highpassfilter(time, flux, width_microhertz
                                  ,sigma_flag=sigma_flag, sigma_width=sigma_clip, hpf_flag=hpf_flag)
    
    if plot_LC:
        fig=plt.figure(figsize=(10,5.5))
        grid=(22,10)
        axlc    = plt.subplot2grid(grid,(0,0),rowspan=10,colspan=10)
        axpsd   = plt.subplot2grid(grid,(12,0),rowspan=10,colspan=10)
        fig.subplots_adjust(bottom=0.1,top=0.98,left=0.15,right=0.97,hspace=5,wspace=0)
        if to_ppm:
            flux_ppm = (flux -1)*1e6
        else:
            flux_ppm = flux
        plot_lightcurve(time, flux_ppm, JD=JD,axlc=axlc)
    elif plot:
        if not axpsd:
            fig=plt.figure()
            axpsd=plt.gca()
            #fig.subplots_adjust(bottom=0.1,top=0.98,left=0.15,right=0.97,hspace=5,wspace=0)
    
    freq,psd=calc_lomb_scargle(time,flux,oversample=oversample)
    #plt.plot(freq,psd)

    if plot:
        if plot_nu_max==True:
            plot_nu_max =[30.0,74.6]
        if plot_nu_max:
            for plot_nu_maxi in plot_nu_max:
                axpsd.axvline(plot_nu_maxi,color='red')
        plot_PSD(freq, psd, axPSD=axpsd,logaxis=logaxis,label=label)
        
    if return_lc:
        return freq,psd,time,flux
    return freq,psd

#%%
#%% theo mixed modes
    
class asympthotic_theory(): # consider jax # use functools.partial?
    #%%% physics
    def __init__(self, pi_modes=None, gamma_modes=None, q=None, dpi_Ms=None, dnu=None
                     ,buoyancy_glitch=False, A_g=0.5, omega_g=100, phi_g=np.pi,Delta_g=0,glitch_type='gaussian'
                     ,acoustic_glitch=False, A_p=0.5, tau_p  =100, phi_p=np.pi,b_p=0
                     ,interp_kind='nearest'):
        
        self.pi_modes=pi_modes
        self.gamma_modes=gamma_modes
        
        self.q=q
        self.dpi_Ms=dpi_Ms
        self.dnu=dnu
        
        self.buoyancy_glitch=buoyancy_glitch
        self.glitch_type=glitch_type
        self.A_g=A_g
        self.omega_g=omega_g
        self.phi_g=phi_g
        self.Delta_g=Delta_g
        
        self.acoustic_glitch=acoustic_glitch
        self.A_p=A_p
        self.tau_p=tau_p
        self.phi_p=phi_p
        self.b_p  =b_p
        
        self.pi_min=np.min(self.pi_modes)
        self.pi_max=np.max(self.pi_modes)
        
        self.gamma_min=np.min(self.gamma_modes)
        self.gamma_max=np.max(self.gamma_modes)
        
        self.gamma_modes_period=np.sort(1/self.gamma_modes)
        
        self.n_p = np.arange(len(self.pi_modes))
        self.n_g = np.arange(len(self.gamma_modes))
        
        if isinstance(q,(np.ndarray,list)):
            self.q_func=interp1d(self.pi_modes,self.q,kind='nearest',bounds_error=False,fill_value=(self.q[0],self.q[-1]))
        else:
            self.q_func = lambda x: self.q
        
    def Theta_P(self,mixed_modes):
        
        if len(self.pi_modes) == 1:
            return np.pi * (mixed_modes - self.pi_modes[0]) / self.dnu
        
        theta_p= np.pi * np.where(
            (mixed_modes <= self.pi_max) & (mixed_modes >= self.pi_min)
            # ,pchip_interpolate(pi_modes, np.arange(len(pi_modes)), mixed_modes)
            ,np.interp(mixed_modes,self.pi_modes,self.n_p)
            ,(mixed_modes - nearest(mixed_modes, self.pi_modes)) / self.dnu + np.round((nearest(mixed_modes,self.pi_modes)-self.pi_modes[0]) / self.dnu)
            )
        # return np.pi * (mixed_modes     - pi_modes)     / s.dnu
        
        if self.acoustic_glitch:
            theta_p -= self.Theta_P_glitch(mixed_modes)

        return theta_p
    
    def Theta_G(self,mixed_modes):
        
        theta_g= np.pi * np.where(
            (mixed_modes <= self.gamma_max) & (mixed_modes >= self.gamma_min)
            # ,-pchip_interpolate(np.sort(1/gamma_modes), np.arange(len(gamma_modes)), 1/mixed_modes)
            ,-np.interp(1/mixed_modes,self.gamma_modes_period,self.n_g)
            
            # ,-((1/mixed_modes - nearest(1/mixed_modes, 1/gamma_modes) ) / dpi_Ms)# + np.round((nearest(1/mixed_modes, 1/gamma_modes) - 1/gamma_modes[0])/ dpi_Ms))
            ,-(np.interp(nearest(1/mixed_modes, 1/self.gamma_modes),self.gamma_modes_period,self.n_g)+(1/mixed_modes-nearest(1/mixed_modes, 1/self.gamma_modes)) / self.dpi_Ms
            )
            )
        # return np.pi * (1.0/self.gamma_modes - 1.0/mixed_modes) / self.dpi_Ms
        
        if self.buoyancy_glitch:
            theta_g += self.Theta_G_glitch(mixed_modes)
        return theta_g
    
    def Theta_P_glitch(self, mixed_modes): #Broomhall2014
        omega = 2*np.pi*mixed_modes

        Phi_p = (self.A_p* np.exp(-2*self.b_p**2 * omega**2)* np.cos(2*(self.tau_p*omega + self.phi_p)))
        return Phi_p
    
    def Theta_G_glitch(self,mixed_modes): #cunha2024
        
        omega = 2*np.pi*mixed_modes
        # P = 2*np.pi/omega
        beta2 = self.omega_g / omega + np.pi/4 + self.phi_g
        
        if self.glitch_type=='gaussian':
            f = 1/ omega * np.exp(-2 * self.Delta_g**2 / omega**2) 
            x = (1/(self.A_g * f * np.sin(beta2)**2) - 1/np.tan(beta2)) 
            
            Phi_glitch = np.arctan(1/x)
            Phi_glitch = np.where(Phi_glitch < 0, Phi_glitch + np.pi, Phi_glitch)
            
        elif self.glitch_type=='step-like':
            x = (-2/(self.A_g * np.sin(2*beta2)) - 1/np.tan(beta2))
            
            Phi_glitch= (np.pi/2) - np.arctan(x)
            Phi_glitch = np.where(Phi_glitch > np.pi / 2, Phi_glitch - np.pi, Phi_glitch)
    
        # beta1 = 2 * self.omega_g / (2*np.pi*mixed_modes) + 2*self.phi_g
        # beta2 = self.omega_g / (2*np.pi*mixed_modes) + np.pi/4 + self.phi_g
        
        # X = -(2 + 2*self.A_g*np.cos(beta2)**2) / (self.A_g*np.cos(beta1))
        #Phi_glitch=np.arctan(1/ X)
    
        # Phi_glitch = np.arctan(1/x)
        # Phi_glitch = np.where(Phi_glitch < 0, Phi_glitch + np.pi, Phi_glitch)
        
        # Phi_glitch= (np.pi/2) - np.arctan(x)
        # Phi_glitch= np.arctan(1/x)
    
        return Phi_glitch#np.arctan(1/ X)
    
    def calc_F(self, mixed_modes,  Θ_p, Θ_g , q):
        return np.tan(Θ_p) * np.tan(Θ_g) - q
    
    def calc_dPhi_acousticglitch(self,mixed_modes):
        omega = 2*np.pi*mixed_modes
    
        phase = 2*(self.tau_p*omega + self.phi_p)
    
        E = np.exp(-2*self.b_p**2 * omega**2)
    
        C = np.cos(phase)
        S = np.sin(phase)
    
        d_Phi =  self.A_p * E * (-8*np.pi* self.b_p**2*omega*C - 4*np.pi*self.tau_p*S)
    
        dd_Phi =  self.A_p * E * (
            ( 64*np.pi**2*self.b_p**4*omega**2
             -16*np.pi**2*self.b_p**2
             -16*np.pi**2*self.tau_p**2) * C
             +64*np.pi**2*self.b_p**2*self.tau_p*omega*S)
    
        return d_Phi, dd_Phi
    
    def calc_dPhi_buoyancyglitch(self, mixed_modes):
        omega = 2 * np.pi * mixed_modes

        b1 = 2 * self.omega_g / omega + 2 * self.phi_g
        b2 = self.omega_g / omega + np.pi/4 + self.phi_g
        db1 = -2 * self.omega_g * (2*np.pi) / omega**2
        db2 = -self.omega_g * (2*np.pi) / omega**2
        
        ddb1 = 4 * self.omega_g * (2*np.pi)**2 / omega**3
        ddb2 = 2 * self.omega_g * (2*np.pi)**2 / omega**3
    
        sin_b1 = np.sin(b1)
        cos_b1 = np.cos(b1)
        cos_b2 = np.cos(b2)
    
        if self.glitch_type == 'step-like':
    
            top = -(2 + 2 * self.A_g * cos_b2**2)
            bot = self.A_g * cos_b1
            X = top / bot
    
            d_top = 2 * self.A_g * np.sin(2 * b2) * db2
            d_bot = -self.A_g * sin_b1 * db1
    
            dX = (d_top * bot - top * d_bot) / (bot**2)
    
            dd_top = 2 * self.A_g * (2 * np.cos(2 * b2) * db2**2 + np.sin(2 * b2) * ddb2)
            dd_bot = -self.A_g * (cos_b1 * db1**2 + sin_b1 * ddb1)
    
            # ddX = ( (dd_top * bot - top * dd_bot) * bot- 2 * d_bot * (d_top * bot - top * d_bot)) / (bot**3)
            ddX = (dd_top * bot - top * dd_bot - 2 * dX * d_bot * bot) / (bot**2)
    
            inv_sq_X = 1 / (1 + X**2)
    
            d_Phi = -inv_sq_X * dX
            dd_Phi = -inv_sq_X * (ddX - 2 * X * dX**2 * inv_sq_X)
    
            return d_Phi, dd_Phi
        
        elif self.glitch_type == 'gaussian':
        
            beta2 = self.omega_g / omega + np.pi/4 + self.phi_g
            dbeta = -(self.omega_g / omega**2) * (2*np.pi)
            ddbeta = 2 * self.omega_g * (2*np.pi)**2 / omega**3
        
            S = np.sin(beta2)
            C = np.cos(beta2)
            dS = C * dbeta
            dC = -S * dbeta
            ddS = -S * dbeta**2 + C * ddbeta
            ddC = -C * dbeta**2 - S * ddbeta
        
            f = (1 / omega) * np.exp(-2 * self.Delta_g**2 / omega**2)
            K = self.A_g * f
        
            df = f * (-1/omega+ (4 * self.Delta_g**2 / omega**3)) * 2 * np.pi
            dK = self.A_g * df
            
            U = K * S**2
            dU = dK * S**2 + K * 2 * S * dS
            ddU = ((self.A_g * (df * 2 * S * dS + f * (2 * dS**2 + 2 * S * ddS))))
        
            X = 1.0 / U  -  C / S
            dX = -dU / (U**2) - (dC * S - C * dS) / (S**2)
            ddX = (2 * dU**2 / U**3) - (ddU / U**2) - ((ddC * S - C * ddS) * S**2 - (dC * S - C * dS) * (2 * S * dS)) / (S**4)
        
            inv = 1 / (1 + X**2)
        
            d_Phi = -inv * dX
            dd_Phi = -inv * (ddX - 2 * X * dX**2 * inv)
        
            return d_Phi, dd_Phi
            
    
    
    def calc_dF(self,mixed_modes,  Θ_p, Θ_g , dq=0, d_Phi_p=0, d_Phi_g=0 ):
        d_th_p = np.pi / self.dnu
        d_th_g = np.pi / (self.dpi_Ms * mixed_modes**2) 
        
        d_th_p = np.pi/self.dnu

        if self.acoustic_glitch:
            d_th_p -= d_Phi_p
        
        if self.buoyancy_glitch:
            d_th_g += d_Phi_g
            
        return ( np.tan(Θ_g) / np.cos(Θ_p)**2 * d_th_p 
               + np.tan(Θ_p) / np.cos(Θ_g)**2 * d_th_g
               -dq)
    
    def calc_ddF(self,mixed_modes,  Θ_p, Θ_g , ddq=0, d_Phi_p=0,dd_Phi_p=0,d_Phi_g=0,dd_Phi_g=0):
        d_th_p = np.pi / self.dnu
        dd_th_p = 0.0 
        d_th_g = np.pi / (self.dpi_Ms * mixed_modes**2)
        dd_th_g = -2 * np.pi / (self.dpi_Ms * mixed_modes**3)
        
        if self.acoustic_glitch:
            d_th_p -= d_Phi_p
            dd_th_p-= dd_Phi_p
        
        if self.buoyancy_glitch:
            d_th_g += d_Phi_g
            dd_th_g += dd_Phi_g
            
        tan_p = np.tan(Θ_p)
        tan_g = np.tan(Θ_g)
        
        sec2_p = 1/np.cos(Θ_p)**2
        sec2_g = 1/np.cos(Θ_g)**2
        
        u_pp_v = tan_g * sec2_p *(2 * tan_p * d_th_p**2 + dd_th_p)
        v_pp_u = tan_p * sec2_g *(2 * tan_g * d_th_g**2 + dd_th_g)
        
        cross  = 2 * (sec2_p * d_th_p) * (sec2_g * d_th_g)
        
        return u_pp_v + v_pp_u + cross - ddq 
    
    # def calc_ζ(mixed_modes, pi_modes, gamma_modes, q, dpi_Ms, dnu): #mosser15 wrong!!
    
    #     Θ_p=Theta_P(mixed_modes, pi_modes, dnu)
    #     Θ_g=Theta_G(mixed_modes, gamma_modes, dpi_Ms)
        
    #     factor = (1.0/q) * ((mixed_modes**2*dpi_Ms)/dnu)
    #     ratio = np.cos(Θ_g)**2 / np.cos(Θ_p)**2
        
    #     zeta = (1 + factor * ratio)**-1
    #     return zeta
    
    def calc_ζp(self,mixed_modes):
        
        q_p = self.q_func(mixed_modes)
        θ_p = self.Theta_P(mixed_modes)
    
        return 1/(1 + self.dpi_Ms / self.dnu * mixed_modes**2 / (q_p * np.cos(θ_p)**2 + np.sin(θ_p)**2/q_p))
    
    
        #%%% halley
    def halley_iteration(self,x, y, dy, ddy, damp=1.):
        
        denom = 2 * dy * dy - y * ddy
        denom = np.where(np.abs(denom) < 1e-15,np.where(denom >= 0, 1e-15, -1e-15),denom)
        step=2 * y * dy / denom
        
        step= np.clip(step,-self.max_step,self.max_step)
        
        step_max = np.max(self.damp*np.abs(step))
        if step_max > np.max([self.prev_step,1e-4]):
            self.damp = max(0.2, self.damp * 0.5)
        else:
            self.damp = min(1.0, self.damp * 1.3)
        
        step= self.damp * step
        
        # step = 2 * y * dy / (2 * dy * dy - y * ddy)
        return x - step, step
    
    def couple_halley(self, maxiter=50,damp=1,tol=1e-5):
        
        self.damp=damp
        self.prev_step = np.inf
        mixed_p_modes  = np.copy(self.pi_modes)
        
        if self.acoustic_glitch:
            mixed_p_modes = calc_glitched_pi_modes(pi_modes=self.pi_modes,dnu=self.dnu,A_p=self.A_p,tau_p=self.tau_p,phi_p=self.phi_p,b_p=self.b_p)
            
            # omega_mid=2*np.pi*np.mean(self.gamma_modes)
            # A_eff = self.A_p * np.exp(-2 * self.b_p**2 * omega_mid**2)
        else: 
            mixed_p_modes = np.copy(self.pi_modes)
        
        if self.buoyancy_glitch:
            mixed_g_modes = calc_gliched_gamma_modes(gamma_modes=self.gamma_modes,dpi_Ms=self.dpi_Ms,A_g=self.A_g,omega_g=self.omega_g,phi_g=self.phi_g,Delta_g=self.Delta_g,glitch_type=self.glitch_type)
        else: 
            mixed_g_modes = np.copy(self.gamma_modes)
        
        self.max_step=np.max([ 0.2* np.min(np.diff(mixed_g_modes)),1e-2])
        
        d_Phi_p, dd_Phi_p = 0.0, 0.0
        d_Phi_g, dd_Phi_g = 0.0, 0.0
            
        for niter in range(maxiter):
        
            q_p = self.q_func(mixed_p_modes)
            q_g = self.q_func(mixed_g_modes)
    
            Θ_p=self.Theta_P(mixed_p_modes)
            Θ_g=self.Theta_G(mixed_p_modes)
                
            if self.acoustic_glitch:
                d_Phi_p, dd_Phi_p = self.calc_dPhi_acousticglitch(mixed_p_modes)
                
            if self.buoyancy_glitch:
                d_Phi_g, dd_Phi_g = self.calc_dPhi_buoyancyglitch(mixed_p_modes)
            
            mixed_p_modes, step_p = self.halley_iteration(mixed_p_modes,
                                self.calc_F(  mixed_p_modes,  Θ_p, Θ_g ,q_p),
                                self.calc_dF( mixed_p_modes,  Θ_p, Θ_g ,d_Phi_p=d_Phi_p,d_Phi_g=d_Phi_g),
                                self.calc_ddF(mixed_p_modes,  Θ_p, Θ_g ,d_Phi_p=d_Phi_p,d_Phi_g=d_Phi_g
                                              ,dd_Phi_p=dd_Phi_p,dd_Phi_g=dd_Phi_g),  damp=self.damp)
            
            Θ_p=self.Theta_P(mixed_g_modes)
            Θ_g=self.Theta_G(mixed_g_modes)
                
            if self.acoustic_glitch:
                d_Phi_p, dd_Phi_p = self.calc_dPhi_acousticglitch(mixed_g_modes)
            
            if self.buoyancy_glitch:
                d_Phi_g, dd_Phi_g = self.calc_dPhi_buoyancyglitch(mixed_g_modes)
            
            mixed_g_modes,step_g = self.halley_iteration(mixed_g_modes,
                                self.calc_F(  mixed_g_modes,  Θ_p, Θ_g , q_g),
                                self.calc_dF( mixed_g_modes,  Θ_p, Θ_g ,d_Phi_p=d_Phi_p,d_Phi_g=d_Phi_g),
                                self.calc_ddF(mixed_g_modes,  Θ_p, Θ_g ,d_Phi_p=d_Phi_p,d_Phi_g=d_Phi_g
                                              ,dd_Phi_p=dd_Phi_p,dd_Phi_g=dd_Phi_g), damp=self.damp)
            
            step_max = max(np.max(np.abs(step_p)),
                           np.max(np.abs(step_g)))
            self.prev_step=step_max
            
            if step_max < tol:
                break
        
        if niter==maxiter-1 and step_max>1e-2:
            print('not converged', niter,self.damp)
        #     if self.buoyancy_glitch:
        #         print('      ',self.A_g,self.omega_g,self.phi_g,step_max)
        #     if self.acoustic_glitch:
        #         print('      ',A_eff,self.A_p,self.tau_p,self.phi_p,self.b_p,step_max)
        
        mixed_modes=np.sort(np.append(mixed_p_modes,mixed_g_modes))
        if np.any(np.diff(mixed_modes) < 1e-5):
            print('missed a mode!!!!!')
            # if self.buoyancy_glitch:
            #     print('      ',self.A_g,self.omega_g,self.phi_g,self.q)
            # if self.acoustic_glitch:
            #     print('      ',A_eff,self.A_p,self.tau_p,self.phi_p,self.b_p,self.q)
        return mixed_p_modes, mixed_g_modes

    #%%% pure modes

def calc_glitched_pi_modes(pi_modes,dnu,eps_p=None,np1=None,A_p=0.05,tau_p=5000,phi_p=np.pi,b_p=0,n_iter=10): 

    if np1 is None:
        np1 = np.round(pi_modes/dnu)
        if eps_p is None:
            eps_p = np.mean(pi_modes/dnu - np1)
            
    for _ in range(n_iter):

        omega = 2*np.pi*pi_modes
        Phi_glitch = (A_p* np.exp(-2*b_p**2*omega**2) * np.cos(2*(tau_p*omega + phi_p)))

        pi_modes_new = ((np1 + eps_p + Phi_glitch/np.pi)*dnu)

        if np.allclose(pi_modes_new, pi_modes):
            break

        pi_modes = pi_modes_new

    return pi_modes

def calc_pi_modes_theo(dnu,eps_p1,fminmax,pi_modes_init=False,interp_kind='linear'
                      ,acoustic_glitch=False,A_p=0.5,tau_p=100,phi_p=np.pi,b_p=1e-9):
    
    np1=np.arange(fminmax[0]//dnu,fminmax[1]//dnu+1)
    
    if isinstance(eps_p1,(list,np.ndarray)) and (isinstance(pi_modes_init,(list,np.ndarray)) and len(pi_modes_init)>1):
            np1_approx=pi_modes_init//dnu
            if interp_kind=='linear':
                eps_p1=np.interp(np1, np1_approx, eps_p1)
            else:
                eps_p1 = interp1d(np1_approx, eps_p1, kind=interp_kind,bounds_error=False, fill_value="extrapolate")(np1)
    
    pi_modes_theo= (np1+eps_p1)*dnu 
    
    if acoustic_glitch:
        pi_modes_theo=calc_glitched_pi_modes(pi_modes=pi_modes_theo,dnu=dnu,eps_p=eps_p1,np1=np1,A_p=A_p,tau_p=tau_p,phi_p=phi_p,b_p=b_p)
        
    pi_modes_theo=pi_modes_theo[(pi_modes_theo>fminmax[0]) & (pi_modes_theo<fminmax[1])]
    
    return pi_modes_theo

def calc_gliched_gamma_modes(gamma_modes,dpi_Ms,eps_g1=None,ng=None
                            ,glitch_type='step-like',A_g=0.5,omega_g=100,phi_g=np.pi,Delta_g=0,n_iter_glitch=10):
    if ng is None:
        ng=1/gamma_modes//dpi_Ms
        eps_g1=1/gamma_modes%dpi_Ms/dpi_Ms
        
    for _ in range(n_iter_glitch):
        omega = 2*np.pi*gamma_modes
        #beta2 = (omega_g_star * P0 / (2*np.pi)+ np.pi/4+ delta)
        beta2 = omega_g / omega + np.pi/4 + phi_g
        
        if glitch_type=='gaussian':
            f = 1/ omega * np.exp(-2 * Delta_g**2 / omega**2) 
            x=(1/(A_g * f * np.sin(beta2)**2) - 1/np.tan(beta2))
            
            Phi_glitch= np.arctan(1/x)
            Phi_glitch = np.where(Phi_glitch < 0, Phi_glitch + np.pi, Phi_glitch)
            
        elif glitch_type=='step-like':
            x=(-2/(A_g * np.sin(2*beta2)) - 1/np.tan(beta2)) 
            
            Phi_glitch= (np.pi/2) - np.arctan(x)
            Phi_glitch = np.where(Phi_glitch > np.pi / 2, Phi_glitch - np.pi, Phi_glitch)
        
        gamma_modes_new=1/((eps_g1+ng+Phi_glitch/np.pi)*dpi_Ms)
        if np.allclose(gamma_modes_new, gamma_modes, rtol=1e-20):
            break
        
        gamma_modes=gamma_modes_new
    
    # plt.figure()
    # plt.plot(gamma_modes,Phi_glitch)
    
    return gamma_modes
        
def calc_gamma_modes_theo(dpi_Ms,eps_g1,freq_minmax,pi_modes=False,interp_kind='linear'
                         ,buoyancy_glitch=False,glitch_type='gaussian',A_g=0.5,omega_g=100,phi_g=np.pi,Delta_g=0,n_iter_glitch=10):
    
    ng_minmax=1/np.array(freq_minmax) //dpi_Ms
    ng=np.arange(ng_minmax[0],ng_minmax[1],-1)
    
    if isinstance(eps_g1,(list,np.ndarray)):
        ng_approx=1/pi_modes //dpi_Ms
        if interp_kind=='linear':
            eps_g1=np.interp(ng, ng_approx[::-1], eps_g1[::-1])
        else:
            eps_g1 = interp1d(ng_approx[::-1], eps_g1[::-1], kind=interp_kind,bounds_error=False, fill_value="extrapolate")(ng)
    
    gamma_modes=1/((eps_g1+ng)*dpi_Ms)
    
    if buoyancy_glitch:
        gamma_modes=calc_gliched_gamma_modes(gamma_modes=gamma_modes,dpi_Ms=dpi_Ms,eps_g1=eps_g1,ng=ng
                                            ,A_g=A_g,omega_g=omega_g,phi_g=phi_g,Delta_g=Delta_g,n_iter_glitch=n_iter_glitch
                                            ,glitch_type=glitch_type)
    
    return gamma_modes

    #%%% streching
def calc_streched_period(freq,q,dnu,dpi,pi_modes=None,eps_p1=None
                        ,acoustic_glitch=False,A_p=0.3,tau_p=1000,phi_p=np.pi,b_p=1e-9):
    if not pi_modes is None:
        Theta_p=asympthotic_theory(pi_modes=pi_modes,dnu=dnu
                                  ,acoustic_glitch=acoustic_glitch,A_p=A_p,tau_p=tau_p,phi_p=phi_p,b_p=b_p).Theta_P(mixed_modes=freq)
    else:
        Theta_p = np.pi * (freq/dnu - eps_p1)
        
    tau           = 1/(freq*1e-6) + dpi/np.pi * np.arctan(q / np.tan(Theta_p))
    return tau

def calc_streched_freq(freq,q,dnu,dpi,gamma_modes=None,eps_g1=None
                      ,buoyancy_glitch=False,glitch_type='gaussian',A_g=1,omega_g=1000,phi_g=np.pi,Delta_g=0):
    if not gamma_modes is None:
        Theta_g=asympthotic_theory(gamma_modes=gamma_modes,dpi_Ms=dpi*1e-6
                                  ,buoyancy_glitch=buoyancy_glitch,glitch_type=glitch_type
                                  ,A_g=A_g,omega_g=omega_g,phi_g=phi_g,Delta_g=Delta_g).Theta_G(freq)
                       
    else:
        Theta_g = np.pi * (eps_g1 -1/(freq*1e-6) /dpi)
    
    
    freq_streched = freq         - dnu/np.pi * np.arctan(q / np.tan(Theta_g))
    return freq_streched


#%%
                 
#%% Star class
class Star():
    #%%%
    def __init__(s,starname='noName',freq=False,psd=False,time=False,flux=False
                  ,dnu=False,numax=False,fminmax=False,dnufactor=5
                  ,lList=[0,2,1],lcolor=['limegreen','dodgerblue',(255/256,171/256,53/256)],lmarker = ['D','o','^','o']
                  ,dnu_rot=False
                  ,star_directory='./Stars/'
                  ,inflateFWHM=False,MCMC_or_dynesty=False,minimizer='minimize',discrete_orders=False,iterations_for_fit=False, fit_dpi=False
                  
                  ,buoyancy_glitch=False,A_g_glitch=2,omega_g_glitch=1000,phi_g_glitch=0.,Delta_g_glitch=0
                  ,acoustic_glitch=False,A_p_glitch=2,tau_p_glitch=1000  ,phi_p_glitch=0.,b_p_glitch=0
                  ,add_glitch_as_phase=False,glitch_type='gaussian', fixed_delta_g_glitch=True
                  
                  ,use_pre_fitted_l1_modes=True
                  ,**kwargs):
        
        s.starname=starname
        
        s.psd_full  = psd
        s.freq_full = freq
        s.psd_raw  = psd
        s.freq_raw = freq
        s.time = time
        s.flux = flux
        s.dnu  = dnu
        s.numax= numax
        
        s.minimizer=minimizer
        s.MCMC_or_dynesty=MCMC_or_dynesty
        s._discrete_orders=discrete_orders
        s.iterations_for_fit=iterations_for_fit
        
        s.buoyancy_glitch=buoyancy_glitch

        s.A_g_glitch=A_g_glitch
        s.omega_g_glitch=omega_g_glitch
        s.phi_g_glitch= phi_g_glitch
        s.Delta_g_glitch=Delta_g_glitch
        
        s.acoustic_glitch=acoustic_glitch
        
        s.A_p_glitch=A_p_glitch
        s.tau_p_glitch=tau_p_glitch
        s.phi_p_glitch= phi_p_glitch
        s.b_p_glitch=b_p_glitch
        
        s.fit_dpi=fit_dpi
        s.use_pre_fitted_l1_modes=use_pre_fitted_l1_modes
        
        if (acoustic_glitch or buoyancy_glitch) and discrete_orders:
            s.discrete_orders=False
            print('set disctrete_orders to False')
            
        s.fixed_delta_g_glitch=fixed_delta_g_glitch
        s.glitch_type=glitch_type
        if buoyancy_glitch:
            if add_glitch_as_phase:
                s.buoyancy_glitch_phase=True
                s.buoyancy_glitch_freq=False
            else:
                s.buoyancy_glitch_phase=False
                s.buoyancy_glitch_freq=True
        
        if acoustic_glitch:
            if add_glitch_as_phase:
                s.acoustic_glitch_phase=True
                s.acoustic_glitch_freq=False
            else:
                s.acoustic_glitch_phase=False
                s.acoustic_glitch_freq=True
        
        if not isinstance(s.time,bool):
            s.T=s.time[-1]-s.time[0]

        s.dnu_rot=dnu_rot
        
        s.inflateFWHM=inflateFWHM
            
        s.fminmax=fminmax
        s.lList=lList
        s.lcolor=lcolor
        s.lmarker=lmarker
        
        s.dnufactor=dnufactor
        s.star_directory='./Stars/'
        
        if not isinstance(s.psd_full,bool):
            s.freq,s.psd=s.set_fminmax(freq=s.freq_full,psd=s.psd_full,fminmax=s.fminmax)
            if not hasattr(s,'y_to_scale'):
                s.y_to_scale=np.max(s.psd)
            
        for k, v in kwargs.items():
            setattr(s, k, v)
            
        if not hasattr(s,'maxiter_halley'):
            s.maxiter_halley=50
        
        if not hasattr(s,'fit_n_dependent_dpi'):
            s.fit_n_dependent_dpi=False
        
    def __getattr__(s, name):
        if name == "freq_sm":
            return s.freq_full
        if name == "psd_sm":
            return s.psd_full
        if name=="freq_minmax_to_Fit":
            print("no freq_minmax_to_Fit found; used fminmax")
            return s.fminmax
        if name== "pi_modes_init" and hasattr(s,"pi_modes_theo"):
            if not s.acoustic_glitch:
                print("no pi_modes_init found; used pi_modes_theo")
            return s.pi_modes_theo
        if name== "eps_p1_init" and hasattr(s,"pi_modes_theo"):
            print("no eps_p1_init found; used eps_p1_theo")
            return s.eps_p1_theo
        raise AttributeError(f"{name} not found")
        
    @property
    def dpi_Ms(s):
        return s.dpi*1e-6
    @dpi_Ms.setter
    def dpi_Ms(s, value):
        print("dpi_Ms is calculated automatically, set s.dpi instead")
        
    @property
    def eps_p1_theo(s):
        return s.pi_modes_theo%s.dnu/s.dnu
    @eps_p1_theo.setter
    def eps_p1_theo(s, value):
        print("eps_p1_theo is calculated automatically, set s.pi_modes_theo instead")
        
    @property
    def discrete_orders(s):
        return s._discrete_orders
    @discrete_orders.setter
    def discrete_orders(s,value):
        s._discrete_orders = value 
        if value:
            s.interp_kind='nearest'
        else:
            s.interp_kind='linear' 
        
    def __repr__(s):
        return f"Star('{s.starname}', νmax={s.numax:.2f}, Δν={s.dnu:.2f})"
        
    
    def smooth(s,smooth_percent=0.03,plot=False,method='gauss',do_bin=False):
        freq_res = np.median(np.diff(s.freq_full))
        smooth_stddev=(s.numax *smooth_percent)/freq_res /100
        s.freq_sm,s.psd_sm = smoothing(freq=s.freq_full,psd=s.psd_full,method=method,plot=plot,do_bin=do_bin,stddev=smooth_stddev)
        s.freq_sm,s.psd_sm = s.set_fminmax(fminmax=s.fminmax)
        
        if hasattr(s, 'modes2'):
            s.es.set_modes_to_0()
    
    def set_fminmax(s,fminmax=False,freq=False,psd=False,double=False,tol=1,dnufactor=False):
        if not isinstance(dnufactor,bool):
            s.dnufactor=dnufactor
        if isinstance(freq,bool):
            freq = s.freq_sm
            psd  = s.psd_sm
        
        if isinstance(fminmax,bool):
            fminmax=[s.numax-s.dnufactor*s.dnu,s.numax+s.dnufactor*s.dnu]
            if double:
                fminmax[1]+=s.dnu
        
        s.fminmax=np.array(fminmax)
        fminmax=np.array(s.fminmax)+np.array([-s.dnu*tol,s.dnu*tol])
            
        psd  =  psd[(freq>fminmax[0]) & (freq<fminmax[1])]
        freq = freq[(freq>fminmax[0]) & (freq<fminmax[1])]
        
        return freq,psd
    
    def save(s, filename=False, directory=False):
        
        if not directory:
            if not os.path.exists(s.star_directory):
                os.mkdir(s.star_directory)
            directory=s.star_directory
            
        if isinstance(filename, bool):
            filename=s.starname+".npz"
        
        dict2save = {}
        for name, val in s.__dict__.items():
            if isinstance(val, Variables):
                variables_data=[]
                variables_type=[]
                for element in val._data:
                    element=table2dict(element)
                    variables_data.append(element)
                    variables_type.append(type(element).__name__)
                
                dict2save[name + "._data"]  = np.array(variables_data, dtype=object)
                dict2save[name + "._names"] = np.array(val.variable_names, dtype=object)
                dict2save[name + "._types"] = np.array(variables_type, dtype=object)

            elif isinstance(val, Table):
                dict2save[name] = np.array([table2dict(val)], dtype=object)
    
            elif isinstance(val, np.ndarray):
                dict2save[name] = val
    
            elif isinstance(val, numbers.Number):
                dict2save[name] = np.array(val, dtype=object)
                
            elif isinstance(val, (str, list, tuple)):
                dict2save[name] = np.array(val, dtype=object)
    
        np.savez(directory+filename, **dict2save)
        print(f"Saved star to {directory+filename}")
    
    @classmethod  
    def load(cls,filename):
        all_data = np.load(filename, allow_pickle=True)
        
        dict2load = {}
        variables_as_dict = {}

        for key in all_data.files:
            if '._data' in key:
                val_name = key.split('._data')[0]
                
                variable_names = all_data[val_name + '._names'].tolist()
                variable_types = all_data[val_name + '._types']
                
                variable_data=[]
                # print(all_data[val_name + '._names'][idx_element])
                for idx_element,elementi in enumerate(all_data[key]):
                    typ = variable_types[idx_element]
                    
                    if typ == "Table" and isinstance(elementi, dict):
                        elementi = dict2table(elementi)
                        
                    elif typ == "ndarray" and not isinstance(elementi, np.ndarray):
                        elementi = np.array(elementi)
                    
                    variable_data.append(elementi)
                
                variables_as_dict[val_name] = Variables(variable_names=variable_names, data=variable_data)
            elif '._names' in key:
                continue
            
            else:
                val = all_data[key]
                if val.shape == ():
                    val = val.item()
                if isinstance(val, np.ndarray) and val.dtype == object:
                    val = outer_tolist(val)
                    val = dict2table(val)
                    
                dict2load[key] = val
            
        star = Star(**dict2load)
        
        for name, variables_obj in variables_as_dict.items():
            variables_obj.s = star
            setattr(star, name, variables_obj)
        
        star.make_Bkg_func()
        if hasattr(star, 'modes'):
            star.got_modes()
        return star
        
    def echelle(s,**kwargs):
        return Echelle(s,**kwargs)
    
    def stacked_psd(s,**kwargs):
        return Stacked_psd(s,**kwargs)
    #%%% 2x2 grid dnu numax
        #%%%% 
    def get_global_seismic_parameters(s,time=False,flux=False,freq=False,psd=False,JD=0
                                       ,initial_guess_numax=30,sigma_clip=False,oversample=1
                                       ,AB=False,author='Kepler',cadence=None
                                       ,exponent_harvey=4,num_harvey=2,num_gauss=1
                                       ,plot=True,smooth_stddev_dnucorr=False
                                       ,HPF_width_microhertz=1/(60*60*24)*1e6/5): #5cyc/day
    
        s.HPF_width_microhertz=HPF_width_microhertz
        s.oversample=oversample
        s.sigma_clip=sigma_clip
        s.smooth_stddev_dnucorr=smooth_stddev_dnucorr
        
        s.num_harvey=num_harvey
        s.num_gauss=num_gauss
        s.exponent_harvey=exponent_harvey
    
        if not isinstance(freq,bool):
            s.freq_raw=freq
            s.psd_raw =psd
        
        if plot:
            if num_gauss==1:
                fig=plt.figure(figsize=(13,8))
                grid=(2,2)
                ax_PSD_log   = plt.subplot2grid(grid,(0,0),rowspan=1,colspan=1)
                ax_PSD       = [plt.subplot2grid(grid,(0,1),rowspan=1,colspan=1)]
                ax_dnu       = [plt.subplot2grid(grid,(1,0),rowspan=1,colspan=1)]
                ax_Echelle   = [plt.subplot2grid(grid,(1,1),rowspan=1,colspan=1)]
                
                fig.subplots_adjust(bottom=0.1,top=0.98,left=0.06,right=0.98,hspace=0.18,wspace=0.18)
            
            elif num_gauss==2:
                fig=plt.figure(figsize=(13,8))
                grid=(3,2)
                ax_PSD_log    = plt.subplot2grid(grid,(0,0),rowspan=1,colspan=1)
                ax_PSD        = plt.subplot2grid(grid,(0,1),rowspan=1,colspan=1)
                ax_dnuA       = plt.subplot2grid(grid,(1,0),rowspan=1,colspan=1)
                ax_dnuB       = plt.subplot2grid(grid,(1,1),rowspan=1,colspan=1)
                ax_EchelleA   = plt.subplot2grid(grid,(2,0),rowspan=1,colspan=1)
                ax_EchelleB   = plt.subplot2grid(grid,(2,1),rowspan=1,colspan=1)
                
                fig.subplots_adjust(bottom=0.1,top=0.98,left=0.06,right=0.98,hspace=0.18,wspace=0.18)
                
                Gridspec=ax_PSD.get_subplotspec().subgridspec(2,1,hspace=0.2)
                ax_PSD.remove()
                ax_PSD = [plt.subplot(Gridspec[0, 0]),plt.subplot(Gridspec[1, 0])]

                ax_dnu=[ax_dnuA,ax_dnuB]
                ax_Echelle=[ax_EchelleA,ax_EchelleB]
            
        else:
            ax_PSD_log=False
            ax_dnu=False
        
        if isinstance(s.time,bool) and isinstance(time,bool) and isinstance(s.freq_raw,bool):
            s.time, s.flux, s.JD, s.time_minmax_sections = get_Mast_lc(starname=s.starname,author=author,cadence=cadence,normalize=True,to_ppm=False
                                          ,plot=False)#,axlc=ax_LC)
        else:
            s.time=time
            s.flux=flux
        
        if isinstance(s.time, bool):
            if not hasattr(s,'T') or not hasattr(s, 'f_nyq'):
                raise Exception('please provide timeseries or T and f_nyq')
        else:
            s.T=s.time[-1]-s.time[0]
        
        if isinstance(s.freq_raw,bool):
            if author=='KEPSEISMIC':
                s.freq_raw=s.time
                s.psd_raw=s.flux
            else:
                s.freq_raw,s.psd_raw,s.time_hpf,s.flux_hpf=calc_psd(s.time,s.flux,hpf_flag=True,sigma_clip=sigma_clip,width_microhertz=HPF_width_microhertz,oversample=oversample,JD=JD,to_ppm=False
                              ,plot=False,return_lc=True
                              )
        if not hasattr(s,'f_nyq'):
            s.f_nyq=get_nyqist_freq(s.time)
            
        s.freq_full,s.psd_full,numax = s.remove_Background(freq=s.freq_raw,psd=s.psd_raw
                               ,plot=plot,axPSD=ax_PSD_log,legend=False
                               ,highpassfiltered=True,return_amp=False
                               ,initial_guess_numax=initial_guess_numax
                               # ,initial_guess=[642.2109554135537,272249,3.,30,103696,30.,4]
                               # ,bounds=(0,np.inf)
                               )
        
        dnu=[0]*len(numax)
        for ABi in range(len(numax)):
            dnu[ABi]=s.get_dnu(numax=numax[ABi],ax_dnu=ax_dnu[ABi],plot=plot,smooth_stddev=smooth_stddev_dnucorr)#,interactive=True)

        # psd_noise_low=s.psd_full[(s.freq_full>max(numax)+6*max(dnu))&(s.freq_full<max(numax)+7*max(dnu))]
        # psd_noise_high=s.psd_full[(s.freq_full<min(numax)-6*min(dnu))&(s.freq_full>min(numax)-8*min(dnu))]
        
        # s.stddev_whitenoise=(np.std(psd_noise_low)+np.std(psd_noise_high))/2
        # ax_PSD_log.fill_between(s.freq_full,s.Bkg_func(s.freq_full)-s.stddev_whitenoise,s.Bkg_func(s.freq_full)+s.stddev_whitenoise,zorder=80,alpha=0.5)
        # ax_PSD[0].fill_between(s.freq_full,-s.stddev_whitenoise,s.stddev_whitenoise,zorder=80,alpha=0.5)
        
        if plot:
            # ax_PSD_log.set_xlim(3,284)
            for ABi in range(len(numax)):
                s.dnu=dnu[ABi]
                s.numax=numax[ABi]
                s.smooth()
               
                plot_PSD(s.freq_full, s.psd_full, axPSD=ax_PSD[ABi],logaxis=False)
                ax_PSD[ABi].set_xlim(numax[ABi]-s.dnufactor*dnu[ABi],numax[ABi]+s.dnufactor*dnu[ABi])
                peak_amp=np.max(s.psd_full[(s.freq_full>numax[ABi]-s.dnufactor*dnu[ABi]) & (s.freq_full<numax[ABi]+s.dnufactor*dnu[ABi])])
                ax_PSD[ABi].set_ylim(0-peak_amp*0.05,peak_amp*1.05)
                
                s.echelle(ax_echelle=ax_Echelle[ABi],double=False)
                s.fminmax=False
                
        if not isinstance(AB,bool):
            s.dnu=dnu[AB]
            s.numax=numax[AB]
            s.smooth()
            
        else:
            s.dnu=dnu[0]
            s.numax=numax[0]
            
        if not isinstance(s.psd_full,bool):
            s.freq,s.psd=s.set_fminmax(freq=s.freq_full,psd=s.psd_full,fminmax=s.fminmax)
            if not hasattr(s,'y_to_scale'):
                s.y_to_scale=np.max(s.psd)
    
        return s
                

        #%%%% background
            #%%%%% functions
    def calc_R(s,freq):
        
        x=np.pi*freq/(2*s.f_nyq)
        R = (np.sin(x) / (x))**2
            
        return R
    
    def Gauss(s,freq,amp,sig,mean):
        G= amp * np.exp(-(freq-mean)**2 / (2*sig**2))
        return G

    def Harvey(s,freq,amp,char_freq,exponent=4,Xi=False):
        if Xi:
            Xi = 2*np.sqrt(2)/np.pi
            amp_term= (Xi *amp**2 / char_freq)
        else:
            amp_term=amp
        # B = (Xi *a0**2 / b0) / (1+(freq/b0)**4)
        H = (amp_term/(1+(freq/char_freq)**exponent))
        return H

    def bkgparams2names(s,params,num_gauss=1,fit_exponent_harvey=False):
            
            if isinstance(fit_exponent_harvey,bool) and fit_exponent_harvey:
                exp_harvey=params[-1]
                params=params[:-1]
            
            white_noise =  params[0]
        
            gauss_params = params[1:num_gauss*3+1]
            harvey_params = params[num_gauss*3+1:]
            
            amp_gauss = np.array(gauss_params[:num_gauss])
            sig_gauss = np.array(gauss_params[num_gauss:num_gauss*2])
            numax = np.array(gauss_params[num_gauss*2:])
            
            num_harvey=len(harvey_params)//2
            amp_harvey = np.array(harvey_params[:num_harvey])
            char_freq_harvey = np.array(harvey_params[num_harvey:])
            
            if fit_exponent_harvey: 
                return white_noise,  amp_gauss,sig_gauss,numax   ,  amp_harvey,char_freq_harvey , exp_harvey
            return     white_noise,  amp_gauss,sig_gauss,numax   ,  amp_harvey,char_freq_harvey

    
    def create_PSD_background_fct(s,num_gauss=1,num_harvey=2,exponent_harvey=4,Xi_harvey=False,return_sep=False):
        def PSD_background(freq,*params):
            #params=white_noise,*amp_gauss,*sig_gauss,*nu_max,*amp_harvey,*char_freq_harvey
            
            if not isinstance(freq,(np.ndarray,list)) or freq.ndim == 0:
                H=np.zeros([num_harvey])
                G=np.zeros([num_gauss])
            else:
                H=np.zeros([num_harvey,len(freq)])
                G=np.zeros([num_gauss,len(freq)])
                
            para_names=s.bkgparams2names(params,num_gauss,fit_exponent_harvey=isinstance(exponent_harvey,bool))
            
            if not isinstance(exponent_harvey,bool):
                white_noise, amp_gauss,sig_gauss,numax, amp_harvey,char_freq_harvey = para_names
                exp_harvey=exponent_harvey
            else: 
                white_noise, amp_gauss,sig_gauss,numax, amp_harvey,char_freq_harvey,exp_harvey = para_names
                
            R=s.calc_R(freq)
            
            for hi in range(num_harvey):
                H[hi] = s.Harvey(freq,amp_harvey[hi],char_freq_harvey[hi],exponent=exp_harvey,Xi=Xi_harvey)
            
            for gi in range(num_gauss):
                G[gi] = s.Gauss(freq,amp=amp_gauss[gi],sig=sig_gauss[gi],mean=numax[gi])
            
            Bkg= white_noise+ R*(np.sum(H,0)+np.sum(G,0))

            if return_sep:
                return Bkg,G,H,R,white_noise
            return Bkg
        
        return PSD_background

            #%%%%% fit psd
    def fit_psd(s,bin_freq=False,bin_psd=False,freq=False,psd=False,SaveToClass=True
               ,plot=False,ax=False,figname='',legend=False
               ,initial_guess_numax=30.,initial_guess=False,bounds=True, highpassfiltered=False):
        
        num_harvey=s.num_harvey
        num_gauss=s.num_gauss
        exponent_harvey=s.exponent_harvey
        
        if isinstance(bin_freq,bool):
            # bin_freq,bin_psd= smoothing(freq,psd,bins=10001,method='savgol',wlength=51,polyorder=1,stddev=20,plot=False,do_bin=True)
            bin_freq,bin_psd= smoothing(freq,psd,bins=10001,method='gauss',wlength=51,polyorder=1,stddev=20,plot=False,do_bin=True)
            # bin_freq,bin_psd=smoothing(freq,psd,bins=int(1e4+1),method='savgol',wlength=51,polyorder=1,stddev=20,plot=plot_all,do_bin=True,log_bins=False)
        
        psd_to_scale=np.max(bin_psd)
        bin_psd=bin_psd/psd_to_scale
        psd=psd/psd_to_scale
        
        if highpassfiltered: 
            bin_freq_mask=bin_freq[bin_freq>s.HPF_width_microhertz] 
            bin_psd_mask=bin_psd[bin_freq>s.HPF_width_microhertz]
    
        background_to_fit=s.create_PSD_background_fct(num_gauss=num_gauss,num_harvey=num_harvey,exponent_harvey=exponent_harvey,Xi_harvey=False,return_sep=False)
        
        if not isinstance(initial_guess_numax,(np.ndarray,list)):
            initial_guess_numax=np.array([initial_guess_numax])
        else:
            initial_guess_numax=np.array(initial_guess_numax)
            
        if isinstance(initial_guess,bool):
            initial_guess_w = np.median(bin_psd[-100:])
            #gauss
            initial_guess_amp_gauss = np.ones(num_gauss) 
            initial_guess_sig_gauss = np.ones(num_gauss) * 0.15* initial_guess_numax
            
            for AB in range(len(initial_guess_numax)):
                
                initial_guess_amp_gauss[AB] = ( #np.max( bin_psd[(bin_freq>initial_guess_numax[AB]*0.8) & (bin_freq<initial_guess_numax[AB]*1.2)])
                                               np.mean(bin_psd[(bin_freq>initial_guess_numax[AB]*0.8) & (bin_freq<initial_guess_numax[AB]*1.2)])
                                               )
            #harvey
            initial_guess_amp_harvey      = [np.median(bin_psd_mask[:1000])
                                             ,np.mean(bin_psd[(bin_freq>initial_guess_numax[AB]*0.8) & (bin_freq<initial_guess_numax[AB]*1.2)])
                                             ,np.mean(bin_psd[(bin_freq>initial_guess_numax[AB]*0.8) & (bin_freq<initial_guess_numax[AB]*1.2)])
                                             ][:num_harvey] * np.array([1, 1, 0.5])[:num_harvey]
           
            
            if num_harvey==2:
                initial_guess_charfreq_harvey = np.array([0.317*initial_guess_numax[0]**0.970,0.948*initial_guess_numax[0]**0.992]) #kallinger2014 table2
            else:
                initial_guess_charfreq_harvey = np.ones(num_harvey) * initial_guess_numax[0]*([0.5,1,1][:num_harvey])
            
            initial_guess_messy= [initial_guess_w
                                 ,initial_guess_amp_gauss,initial_guess_sig_gauss,initial_guess_numax
                                 ,initial_guess_amp_harvey,initial_guess_charfreq_harvey]
            
            if isinstance(exponent_harvey,bool):
                initial_guess_messy.append(4)
            
        else:
            initial_guess_messy=s.bkgparams2names(initial_guess,fit_exponent_harvey=isinstance(exponent_harvey, bool))
            if isinstance(exponent_harvey,bool):
                initial_guess_w,initial_guess_amp_gauss,initial_guess_sig_gauss,initial_guess_numax,initial_guess_amp_harvey,initial_guess_charfreq_harvey,initial_guess_exponent=initial_guess_messy
            else:
                initial_guess_w,initial_guess_amp_gauss,initial_guess_sig_gauss,initial_guess_numax,initial_guess_amp_harvey,initial_guess_charfreq_harvey=initial_guess_messy
            
            initial_guess_w         =initial_guess_w         /psd_to_scale
            initial_guess_amp_gauss =initial_guess_amp_gauss /psd_to_scale
            initial_guess_amp_harvey=initial_guess_amp_harvey/psd_to_scale
            
            initial_guess_messy= [initial_guess_w
                                 ,initial_guess_amp_gauss,initial_guess_sig_gauss,initial_guess_numax
                                 ,initial_guess_amp_harvey,initial_guess_charfreq_harvey,initial_guess_exponent]
        
        initial_guess=[]
        for sublist in initial_guess_messy:
            if isinstance(sublist,(np.ndarray,list)):
                initial_guess.extend(sublist)
            else:
                initial_guess.append(sublist)
                
        if isinstance(bounds,bool) and bounds:
            bounds_messy= (             
                    (initial_guess_w*0   , initial_guess_amp_gauss*0.1      ,initial_guess_sig_gauss*0.5   ,initial_guess_numax*0.7,  initial_guess_amp_harvey*0.2 ,  initial_guess_charfreq_harvey*0.2  ) 
                   ,(initial_guess_w*1.1 , initial_guess_amp_gauss*1e2      ,initial_guess_sig_gauss*3     ,initial_guess_numax*1.3,  initial_guess_amp_harvey*10 ,  initial_guess_charfreq_harvey*0+1000)              
                    )
            
            lower_bounds=[]
            upper_bounds=[]
            for lower,upper in zip(bounds_messy[0],bounds_messy[1]):
                if isinstance(upper,(np.ndarray,list)):
                    lower_bounds.extend(lower)
                    upper_bounds.extend(upper)
                else:
                    lower_bounds.append(lower)
                    upper_bounds.append(upper)
            bounds=(lower_bounds,upper_bounds)
                
            if isinstance(exponent_harvey,bool):
                bounds[0].append(1.5)
                bounds[1].append(6)
                
        if isinstance(bounds, tuple) and len(bounds) == 2:
            bounds = list(zip(bounds[0], bounds[1]))
        
        if s.minimizer=='minimize':
            result = minimize(fun=neg_log_likelihood_X22dof,args=(bin_freq_mask, bin_psd_mask, background_to_fit)
                             ,bounds=bounds,x0=initial_guess,method='L-BFGS-B' )
        elif s.minimizer=='differential_evolution':
            result = differential_evolution(neg_log_likelihood_X22dof,args=(bin_freq_mask, bin_psd_mask, background_to_fit)
                             ,bounds=bounds,maxiter=1000000,popsize=200)
            print("background fit succeded: ",result.success)
        elif s.minimizer=='basinhopping':
            result = basinhopping(neg_log_likelihood_X22dof, initial_guess  #,minimizer_kwargs=minimizer_options
                                    # ,niter=10000
                                    )
        else:
            raise Exception("only 'minimize', 'basinhopping' or 'differential_evolution' implemented")
    
        parameters = result.x
        
        background_fit=s.create_PSD_background_fct(num_gauss=num_gauss,num_harvey=num_harvey,exponent_harvey=exponent_harvey,Xi_harvey=False,return_sep=True)
        
        para_names = s.bkgparams2names(parameters,num_gauss=num_gauss,fit_exponent_harvey=isinstance(exponent_harvey,bool))
        if not isinstance(exponent_harvey,bool):
            white_noise, amp_gauss,sig_gauss,numax, amp_harvey,char_freq_harvey = para_names
            remove_last_para=True
        else: 
            white_noise, amp_gauss,sig_gauss,numax, amp_harvey,char_freq_harvey,exponent_harvey = para_names
            print("exponent = ",exponent_harvey)
            remove_last_para=False
            
            
        psd*=psd_to_scale
        bin_psd*=psd_to_scale
        bin_psd_mask*=psd_to_scale
            
        white_noise*=psd_to_scale
        amp_gauss*=psd_to_scale
        amp_harvey*=psd_to_scale
        
        bkg_parameters=np.concatenate((np.ravel(white_noise), np.ravel(amp_gauss),np.ravel(sig_gauss),np.ravel(numax), np.ravel(amp_harvey),np.ravel(char_freq_harvey),np.ravel(exponent_harvey)))
        if remove_last_para:
            bkg_parameters=bkg_parameters[:-1]
            
        if SaveToClass:
            s.bkg_parameters=bkg_parameters
        
        Bkg,G,H,R,white_noise=background_fit(bin_freq_mask,*bkg_parameters)
            
        if plot:
            if not ax:
                plt.figure()
                ax=plt.gca()
                plt.tight_layout()
                
            ax.set_xscale('log')
            ax.set_yscale('log')
            
            ax.set_xlabel('ν (μHz)')
            ax.set_ylabel('PSD (ppm² μHz$^{-1}$)')
                
            powerpointblue= (36/259, 49/259, 88/259)
            lightblue=(195/259,209/256,234/256)
            darkblue=(8 /256,31/256,92/256)
            colorpsd=lightblue
            ax.plot(freq,psd,color=colorpsd,alpha=0.6)#'lightgrey')
            ax.plot(bin_freq_mask,bin_psd_mask,'k')
            
            ax.plot(freq,psd,color='lightgrey',zorder=-5)
                
            color_numax='goldenrod'
            color_numax=(255/256,161/256,17/256)
            
            ax.plot(bin_freq_mask,Bkg,label=r'P$_\mathrm{fit}$',color='red',zorder=20,linewidth=2)
            ax.plot(bin_freq_mask,(Bkg-R*(np.sum(G,0))),':',color='red',zorder=20,linewidth=2,label=r'P$_\mathrm{bkg}$')
    
            ax.plot(bin_freq_mask,np.ones_like(bin_freq_mask)*white_noise,'--',label='W',color='orange')#'aquamarine')
            
            colorhi=['darkblue','blue','royalblue']
            for hi in range(num_harvey):  
                H = s.Harvey(bin_freq_mask,amp_harvey[hi],char_freq_harvey[hi],exponent=exponent_harvey)
                ax.plot(bin_freq_mask,H,'--',label=r'H$_{}$'.format(hi),color=colorhi[hi])
                
            colorgi=['hotpink','fuchsia']
            for gi in range(num_gauss):  
                G = s.Gauss(bin_freq_mask,amp_gauss[gi],sig_gauss[gi],numax[gi])
                ax.plot(bin_freq_mask,G,'--',label=r'G$_{}$'.format(gi),color=colorgi[gi])
                
            for numaxi in numax:
                ax.axvline(numaxi,color=color_numax,alpha=0.7,linestyle='-')
                ax.text(numaxi*1.1,np.max(s.psd_raw)*0.5,'ν$_{{max}}$ = {:.2f} μHz'.format(numaxi))#,transform=ax.transAxes)
            # ax.plot(bin_freq,B0,'--',label=r'B$_0$',color='darkblue')
            # if not simple_fit:
            #     ax.plot(bin_freq,B1,'--',label=r'B$_1$',color='blue')
            # if not used_highpassfilter:
            #     ax.plot(bin_freq,B2,'--',label=r'B$_2$',color='royalblue')
            
            # ax.plot(bin_freq,G,'-.',label=r'G$_A$',color='hotpink')
        
            # ax.axvline(nu_max_B,color=color_numax,alpha=0.7,linestyle='-')
            
            # ax.axhline(parameters[5],color=color_numax,alpha=0.7,linestyle='-')
            # ax.axhline(parameters[7],color=color_numax,alpha=0.7,linestyle='-')
    
            if highpassfiltered:
                xmin=s.HPF_width_microhertz*0.7
            else:
                xmin=None
                
            ax.set_xlim(xmin,s.f_nyq*1.01)
            
            ax.set_ylim(np.median(s.psd_raw[-100:])*0.2,np.max(s.psd_raw)*1.3)

            if legend:         
                ax.legend(loc='lower left')
        
        return numax
            #%%%%% rm background
    
    def make_Bkg_func(s):
        
        background_fit=s.create_PSD_background_fct(num_gauss=s.num_gauss,num_harvey=s.num_harvey,exponent_harvey=s.exponent_harvey,Xi_harvey=False,return_sep=True)
        
        def Bkg_func(freq):
            Fit, G, H, R, white_noise = background_fit(np.array(freq), *s.bkg_parameters)
            return (Fit-R*np.sum(G,0))
        
        s.Bkg_func=make_func_accept_Variables(Bkg_func)
        
    
    def remove_Background(s,freq=False,psd=False
                         ,plot=True,axPSD=False,legend=False
                         ,highpassfiltered=False,return_amp=False
                         ,initial_guess=False,initial_guess_numax=False,bounds=True):
    
        if axPSD:
            plot=True
        
        numax=s.fit_psd(freq=freq,psd=psd,plot=plot,ax=axPSD,legend=legend
                       ,highpassfiltered=highpassfiltered
                       ,initial_guess=initial_guess,initial_guess_numax=initial_guess_numax
                       ,bounds=bounds,SaveToClass=True)
        
        s.make_Bkg_func()
        
        Bkg=s.Bkg_func(freq)
        psd_noBack = psd-Bkg
        s.SNR = psd/Bkg
    
        return freq,psd_noBack,numax
    
        #%%%% get dnu
    def get_dnu(s,freq=False,psd=False,numax=False,smooth_stddev=True
                 ,dnu_guess_minmax=False,dnu_initial_guess=False
                 ,fminmax=False
                 ,yminmax=False,xminmax=False
                 ,interactive=False
                 ,plot=True,ax_dnu=False):
        
        if isinstance(freq, bool):
            freq=s.freq_full.copy()
            psd =s.psd_full.copy()
        if not numax:
            numax=s.numax
            
        if isinstance(smooth_stddev,bool) and smooth_stddev:
            smooth_percent = 0.1
            freq_res = np.median(np.diff(freq))
            smooth_stddev=(numax *smooth_percent)/freq_res /100
            
        if not isinstance(smooth_stddev,bool):
            freq,psd_sm = smoothing(freq=freq,psd=psd,method='gauss',plot=False,do_bin=False,stddev=smooth_stddev)
            #,bins=5001,wlength=101,polyorder=2,stddev=10,boxw=6,do_bin=False):
        else:
            psd_sm=psd
        
        if not fminmax:
            fminmax= [numax*0.6,numax*1.4]
        
        mask = (freq > fminmax[0]) & (freq < fminmax[1])
        
        if plot and not ax_dnu:
            fig_dnu=plt.figure(figsize=(10,9))
            grid=(2,1)
            axPSD    = plt.subplot2grid(grid,(0,0),rowspan=1,colspan=1)
            axCorr   = plt.subplot2grid(grid,(1,0),rowspan=1,colspan=1)
                
            axPSD.plot(freq[mask],psd[mask],color='lightgrey')
            plot_PSD(freq[mask],psd_sm[mask],axPSD=axPSD,nu_max=numax
                     ,color_nu_max_line='r')
            
        elif plot and ax_dnu:
            axCorr = ax_dnu
        
        corr = np.correlate(psd_sm[mask],psd_sm[mask],'full') 
        corr = corr[len(corr)//2:]
        corr = corr/corr[0]
        freq_corr =np.linspace(0,len(corr)-1,len(corr)) * np.median(np.diff(freq))
        
        if plot:
            axCorr.plot(freq_corr,corr,'k')
            axCorr.set_xlabel('Δν [μHz]')
            axCorr.set_ylabel('autocorrelation')
        
        if interactive:    
            fig_dnu.canvas.mpl_connect('button_press_event', lambda event: s._onclick_LorentzFit(event,limit_y=True,fit_dnu=True))
            fig_dnu.canvas.mpl_connect('key_press_event', s._onclick_delete_LorentzFit)
          
        else:
            initial_guess_dnu=0.266*numax**0.767
            
            idx_peaks,_=find_peaks(corr, height=np.median(corr)*2, distance=len(freq_corr[freq_corr<0.3*initial_guess_dnu]))
            
            idx_peaks_sort=np.argsort(corr[idx_peaks])[::-1] 
            idx_peaks=idx_peaks[idx_peaks_sort]
            idx_peak_dnu=idx_peaks[(freq_corr[idx_peaks]>0.6*initial_guess_dnu) & (freq_corr[idx_peaks]<1.4*initial_guess_dnu)][0]
            
            # axCorr.axvline(initial_guess_dnu,color='m')
            # for i in idx_peaks:
            #     axCorr.axvline(freq_corr[i])
            # axCorr.axvline(freq_corr[idx_peak_dnu],zorder=1000,color='lime')
            
            idx_min=np.where(corr==np.min(corr[(freq_corr<freq_corr[idx_peak_dnu]) & (freq_corr>numax/20)]))[0][0]
            if isinstance(xminmax, bool):
                xminmax=[freq_corr[idx_min],freq_corr[idx_peak_dnu]+(freq_corr[idx_peak_dnu]-freq_corr[idx_min])]
                
            if isinstance(yminmax, bool):
                yminmax=[corr[idx_min],corr[idx_peak_dnu]*1.1]
                
            if plot:
                axCorr.plot(freq_corr[idx_peak_dnu],corr[idx_peak_dnu],'rx')
                axBox     = axCorr.inset_axes([0.45 , 0.4, 0.53, 0.55])
            else:
                axCorr=False
                axBox=False
                
            dnu=s._fit_Lorenzian_in_box(x=freq_corr,y=corr,xminmax=xminmax,yminmax=yminmax,ax=axCorr,axBox=axBox).x_0.value
            
            if plot:
                axBox.text(0.65,0.87,'Δν = {:.2f} μHz'.format(dnu),transform=axBox.transAxes)
            
            return dnu
        
    
    #%%% get uncertainties for numax dnu
    
    def split8repeat(s,n_steps=None,window_width=None,step=None
                     ,split_into_quarters=False,merge_n_quarters=1
                     ,plot=True):
        
        if not split_into_quarters:
            
            if window_width is None:
                window_width=s.T/n_steps
                step=window_width
            
            elif n_steps is None:
                #step = window_width for not overlaping windows
                if step is None:
                    step=window_width
                n_steps=int((np.max(s.time)-np.min(s.time)-window_width)/step)+1
                
        else:
            if not hasattr(s,'time_minmax_section'):
                print('load lightcurve to get quarte/sector info. please define custom time_minmax instead')
                if 'KIC' in s.starname:
                    author='Kepler'
                elif 'TIC' in s.starname:
                    author='TESS'
                _,_,_, s.time_minmax_sections = get_Mast_lc(starname=s.starname,author=author,normalize=True,to_ppm=False
                                              ,plot=False)
            
            time_minmax_sections=s.time_minmax_sections
            
            if time_minmax_sections[0][1]-time_minmax_sections[0][0]<20:
                time_minmax_sections[1][0]=time_minmax_sections[0][0]
                time_minmax_sections=time_minmax_sections[1:]
                
            if len(time_minmax_sections)%2==1:
                time_minmax_sections[1][0]=time_minmax_sections[0][0]
                time_minmax_sections=time_minmax_sections[1:]
                
            merged_time_minmax = np.column_stack([time_minmax_sections[                  ::merge_n_quarters, 0]
                                                 ,time_minmax_sections[merge_n_quarters-1::merge_n_quarters, 1]])
            n_steps=len(merged_time_minmax)
            
        if plot:
            axlc=generate_subplot_grid((1,1))
        
        if plot:
            axPSDs=generate_subplot_grid((n_steps,1),share_x=True)
            
        numaxs= np.zeros(n_steps)
        dnus  = np.zeros(n_steps)
        for idx_step in range(n_steps):
            if split_into_quarters:
                window_mask=(s.time>merged_time_minmax[idx_step][0]) & (s.time<merged_time_minmax[idx_step][1])
            else:
                window_mask=(s.time>np.min(s.time)+idx_step*step) & (s.time<np.min(s.time)+idx_step*step+window_width)
                
            timei=s.time[window_mask]
            fluxi=s.flux[window_mask]
            
            freqi,psdi,timei_hpf,fluxi_hpf=calc_psd(timei,fluxi,hpf_flag=True,sigma_clip=s.sigma_clip,width_microhertz=s.HPF_width_microhertz,oversample=s.oversample,JD=s.JD,to_ppm=False
                          ,plot=False,return_lc=True)
            
            if plot:
                axlc.plot(timei_hpf,fluxi_hpf)
                
            initial_guess=s.bkg_parameters
            numaxi=s.fit_psd(freq=freqi,psd=psdi
                           ,plot=plot,ax=axPSDs[idx_step]
                           ,highpassfiltered=True
                           ,initial_guess=initial_guess#,bounds=bounds
                           ,SaveToClass=False)

            dnui=s.get_dnu(freq=freqi,psd=psdi,numax=numaxi,plot=plot,smooth_stddev=s.smooth_stddev_dnucorr)#,interactive=True)

            dnus[idx_step]=dnui
            numaxs[idx_step]=numaxi
            
        s.numax_uncertainty=np.std(numaxs)
        s.dnu_uncertainty=np.std(dnus)
    
    #%%% interactive LorentzFit
    
    def _onclick_LorentzFit(s,event,limit_y=False,fit_dnu=False):
        
        if event.inaxes == None:
            return
        elif hasattr(s,'ax_FitLor') and not event.inaxes==s.ax_FitLor:
            del s.xi,s.yi
      
        s.ax_FitLor= event.inaxes
        fig=event.canvas.figure
        
        if not hasattr(s,'xi'):
            s.xi=[]
            s.yi=[]
        
        if not hasattr(s,'fitted_lor'):
            s.fitted_lor=Table(names=('amp_obs','amp_fit','x_0','fwhm','xmin','xmax','sig'))
            
        if len(s.xi)==2:
            s.xi=[]
            s.yi=[]
            
        if len(s.xi)==0:
            s.xi.append(event.xdata)
            s.ax_FitLor.axvline(s.xi[0],color='lightgrey')
            if not limit_y:
                s.yi.append(-np.inf)
            else:
                s.yi.append(event.ydata)
                s.ax_FitLor.axhline(s.yi[0],color='lightgrey')
            
            s.deleted_last_freq=True
            fig.canvas.draw_idle()
            
            
        elif len(s.xi)==1:
            s.xi.append(event.xdata)
            s.ax_FitLor.axvline(s.xi[1],color='lightgrey')
            if not limit_y:
                s.yi.append(np.inf)
            else:
                s.yi.append(event.ydata)
                s.ax_FitLor.axhline(s.yi[1],color='lightgrey')
                
            
            line=s.ax_FitLor.lines[0]
            
            x=line.get_xdata()
            y=line.get_ydata()
            
            if limit_y:
                s.axBox     = s.ax_FitLor.inset_axes([0.45 , 0.4, 0.53, 0.55])
            else:
                s.axBox=False
            
            lorentz=s._fit_Lorenzian_in_box(x,y,xminmax=s.xi,yminmax=s.yi,ax=s.ax_FitLor,axBox=s.axBox,X2dof=~fit_dnu,addBkg=~fit_dnu,scale_y=True)
            
            if fit_dnu:
                s.dnu=lorentz.x_0.value
                s.axBox.text(0.7,0.87,'Δν = {:.2f} μHz'.format(s.dnu),transform=s.axBox.transAxes)
            else:
                
                mask_full =(s.freq_full > s.xi[0]) & (s.freq_full < s.xi[1])
                s.sig=s.get_sig_peakbag(lorentz)
                lorentz_tab=Table(rows=[[np.max(s.psd_full[mask_full]),*lorentz.parameters,s.xi[0],s.xi[1],s.sig]]
                                 ,names=('amp_obs','amp_fit','x_0','fwhm','xmin','xmax','sig'))
    
                s.fitted_lor=vstack([s.fitted_lor, lorentz_tab])
                
                # yerr_pos=s.ax_FitLor.get_ylim()[0]+0.9*(s.ax_FitLor.get_ylim()[1]-s.ax_FitLor.get_ylim()[0])
                # s.ax_FitLor.errorbar(lorentz_tab['x_0'],yerr_pos,xerr=lorentz_tab['std'])
                # s.ax_FitLor.errorbar(lorentz_tab['x_0'],yerr_pos,xerr=lorentz_tab['sig'])
            
            s.deleted_last_freq=False
            fig.canvas.draw_idle()
            
    
    def _onclick_delete_LorentzFit(s,event):
      
        ax= event.inaxes
        if not ax.get_figure() == s.ax_FitLor.get_figure():
            return
        
        line=ax.lines[0]
        
        if hasattr(s,'axBox') and not isinstance(s.axBox,bool) and s.axBox.get_figure() == ax.get_figure():
            s.axBox.remove()
            
            for line in ax.get_lines()[-6:]:
                line.remove()
            
            plt.draw()
        else:
            None
        
        if not s.deleted_last_freq:
            s.fitted_lor=s.fitted_lor[:-1]
            s.deleted_last_freq=True
        else:
            s.deleted_last_freq=False
        
        num_lines_to_delete=1
        if not len(s.xi)==0:
            if s.yi[0]==0:
                if len(s.xi)==2:
                    num_lines_to_delete=3
            else:
                num_lines_to_delete=2 
        else:
            if s.deleted_last_freq:
                num_lines_to_delete=3
            else:
                num_lines_to_delete=1
         
        s.xi = s.xi[:-1]
        s.yi = s.yi[:-1]
            
        for line in ax.get_lines()[-num_lines_to_delete:]:
            line.remove()
        plt.draw()
        
    def _fit_Lorenzian_in_box(s,x,y,xminmax,yminmax=[-np.inf,np.inf],ax=False,axBox=False
                               ,X2dof=False,addBkg=False,scale_y=False
                               ,fwhm_initial_guess=False,H_initial_guess=False
                               ,MCMC=False):
        
        mask_ToFit =(x > xminmax[0]) & (x < xminmax[1]) & (y > yminmax[0]) & (y < yminmax[1])
        if np.isinf(yminmax[0]):
            yminmax[0]=0
            
        if scale_y:
            y_to_scale=np.max(y[mask_ToFit])
            y=y/y_to_scale
            yminmax[0]=yminmax[0]/y_to_scale
        else:
            y_to_scale=1
        
        if addBkg:
            bkg=s.Bkg_func(x[mask_ToFit])/y_to_scale
            Bkg_model=custom_model(lambda x: s.Bkg_func(x)/y_to_scale)()
            ZO=0
        
        else:
            bkg=0
            ZO=1
            
        if not fwhm_initial_guess:
            fwhm_initial_guess=(xminmax[1]-xminmax[0])/4
        if not H_initial_guess:
            H_initial_guess=np.mean(y[mask_ToFit]-yminmax[0]*ZO)
                                  
        l_init = astropy_models.Lorentz1D( x_0=(xminmax[1]-xminmax[0])/2+xminmax[0]
                                  ,fwhm=fwhm_initial_guess
                                  ,amplitude=H_initial_guess
                                  ,bounds={"fwhm": (1e-9, xminmax[1]-xminmax[0])
                                           ,"amplitude":(1e-9,max(s.psd_full)/y_to_scale)
                                           ,"x_0": (xminmax[0],xminmax[1])
                                           })
        
        if addBkg:
            l_init=l_init+Bkg_model
        
        if X2dof:
            if MCMC:
                fit_l = Chi2TwoDofFitter(MCMC=MCMC,labels=['fwhm','amp','x0'])
            else:
                fit_l = Chi2TwoDofFitter(minimizer=s.minimizer)
        else:
            fit_l = fitting.TRFLSQFitter()
        
        lorentz = fit_l(model=l_init, x=x[mask_ToFit],y=y[mask_ToFit]-yminmax[0]*ZO+bkg)
        
        if addBkg:
            lorentz=lorentz[0]
        
        y          *= y_to_scale
        bkg        *= y_to_scale
        yminmax[0] *= y_to_scale
        lorentz.amplitude *= y_to_scale
        
        x_fit=np.linspace(xminmax[0],xminmax[1],10000)
        y_fit=lorentz(x_fit)+yminmax[0]*ZO
        
        # if addBkg:
        #     y_fit=y_fit-s.Bkg_func(x_fit)
        
        x0=lorentz.x_0.value
        if ax:
            if hasattr(s,'spsd') and not axBox:
                try:
                    color=s.lcolor[s.spsd.idx_l]
                except:
                    color='r'
            else:
                color='r'
            ax.plot(x_fit,y_fit,color=color) 
            ax.axvline(x0,color=color)
            if axBox:
                ax.axvline(xminmax[1],color='lightgrey')
                ax.axhline(yminmax[1],color='lightgrey')
                
                ax.axvline(xminmax[0],color='lightgrey')
                ax.axhline(yminmax[0],color='lightgrey')
                
                axBox.plot(x,y,'k')
                axBox.plot(x_fit,y_fit,'r')
                axBox.set_xlim(xminmax[0],xminmax[1])
                axBox.set_ylim(yminmax[0],yminmax[1])
                axBox.set_xlabel('Δν [μHz]')
                axBox.set_ylabel('autocorrelation')
        
        return lorentz
    
    def get_sig_peakbag(s,lorentz):
        d2sec=24*60*60
        
        def min_U(T,tau): #in d
            return 1/(2*np.pi)*np.sqrt(1/(T*d2sec*tau*d2sec))*1e6
        
        def Lib_U(H,Noise,tau,T):
            return min_U(T, tau)*np.sqrt(np.sqrt(1+Noise/H)*(np.sqrt(1+Noise/H)+np.sqrt(Noise/H))**3)
        
        x0  = lorentz.x_0.value         #mHz
        fwhm= lorentz.fwhm.value        #mHz
        H   = lorentz.amplitude.value   #ppm2/mHz
    
        tau = 1/(fwhm*np.pi) /(d2sec*1e-6)  #d
        
        Noise=s.Bkg_func(x0)
        
        sig_Noise_Lib= Lib_U(H,Noise,s.T,tau)
        
        return sig_Noise_Lib
    
    #%%% epsilon
    def calc_epsilon(s,x,mode_type='p',l1=False):
        match mode_type:
            case 'p':
                deltaX=s.dnu 
            case 'g':
                deltaX=s.dpi
    
        x = np.array(x)
        epsilon = x%deltaX/deltaX
        
        a=np.log(5)/0.45
        b=np.log(4)-a
        
        if mode_type=='p':
            eps_theo=(np.log(s.dnu)-b)/a
            if l1:
                eps_theo-=1/2
            
            if isinstance(epsilon,(list,np.ndarray)):
                idx_we,=np.where(np.abs(epsilon-eps_theo)>0.3)
                epsilon[idx_we]+=1
            elif abs(epsilon-eps_theo)>0.3:
                epsilon+=1
        
        return epsilon
    
    #%%% modes
    
    def set_modes(s,modes=False,modes_lor=False,modes0=False,modes1=False,modes2=False
                    ,modes0_lor=False,modes1_lor=False,modes2_lor=False):
        
        if any([modes_lor,modes0_lor,modes1_lor,modes2_lor]):
            if isinstance(modes_lor,bool) and any([modes0_lor,modes1_lor,modes2_lor]):
                s.fitted_lor=Table(names=('amp_obs','amp_fit','x_0','fwhm','xmin','xmax','sig'))
                
                emptyT= Table([[],[],[],[],[],[],[]],names=['amp_obs','amp_fit','x_0','fwhm','xmin','xmax','sig'])
                names_lor=['modes0_lor','modes1_lor','modes2_lor']
                names=['modes0','modes1','modes2']
                names_sig=['modes0_sig','modes1_sig','modes2_sig']
                
                modes_lor=[modes0_lor or emptyT,modes1_lor or emptyT,modes2_lor or emptyT]
                for modesl_lor,name_lor in zip(modes_lor,names_lor):
                    setattr(s, name_lor, modesl_lor)
                    
            elif not isinstance(modes_lor, bool):
                for l, modesl_lor in enumerate(modes_lor):
                    setattr(s, f"modes{l}_lor", modesl_lor)
                names_lor = [f"modes{l}_lor" for l in range(len(modes_lor))]
                names     = [f"modes{l}"     for l in range(len(modes))]
                names_sig = [f"modes{l}_sig"     for l in range(len(modes))]
            
            s.modes_lor = Variables(s=s, variable_names=names_lor)
            modes    =s.modes_lor['x_0']
            modes_sig=s.modes_lor['sig']
            
            for l, modesl_sig in enumerate(modes_sig):
                setattr(s, f"modes{l}_sig", modesl_sig)
            s.modes_sig = Variables(s=s, variable_names=names_sig)
            
        if isinstance(modes,bool) and any([modes0,modes1,modes2]):
            names=['modes0','modes1','modes2']
            modes=[np.array(modes0) or [],np.array(modes1) or [],np.array(modes2) or []]
            for modesl,name in zip(modes,names):
                setattr(s, name, modesl)
                
        elif not isinstance(modes, bool):
            for l, modesl in enumerate(modes):
                setattr(s, f"modes{l}", modesl)
            names = [f"modes{l}" for l in range(len(modes))]
           
        s.modes = Variables(s=s, variable_names=names)

        s.got_modes()
    
    def got_modes(s):

        if hasattr(s, 'modes1'):
            s.get_np1()
            
            if hasattr(s,'es'):
                if hasattr(s, 'dpi') and abs(s.dpi - 100) < 1e-20 and hasattr(s.es, 'boxes') :
                    s.dpi = np.abs(np.median(np.diff(1 / s.modes1)) * 1e6)
                    idx_dpi = np.where(np.array(s.es.variable_names) == 'dpi')[0][0]
                    s.es.boxes[idx_dpi].set_val(str(round(s.variables[1], 3)))

        if hasattr(s,'es') and hasattr(s, 'modes2'):
            s.es.set_modes_to_0()
        
        if not hasattr(s,'fwhm_l0_func') and hasattr(s,'_joint_parameter_result_l0_FWHM'):
            s.fwhm_l0_func= lambda x: slope(x,*s._joint_parameter_result_l0_FWHM)
            
    def get_np1(s):
        tol_next_order = 0.5 * s.dnu
        s.np1 = np.zeros_like(s.modes1, dtype=int)
        if s.modes1[-1] > s.modes0[-1]:
            np.append(s.np1,0)
        
        n_p=s.modes1[0]//s.dnu
        s.np1[0]=np.array(n_p)
        
        for idx_mode in range(1, len(s.modes1)):
            if (np.abs(s.modes1[idx_mode] - s.modes1[idx_mode-1]) > tol_next_order) or (any((s.modes0>s.modes1[idx_mode-1]) & (s.modes0<s.modes1[idx_mode]))): 
                n_p_new=s.modes1[idx_mode]//s.dnu
                if n_p == n_p_new:
                    n_p_new+=1
                n_p=n_p_new
            s.np1[idx_mode] = n_p

    #%%% get theo modes
    def calc_theo_mixed_modes_discrete(s):
        s.mixed_modes_theo_n=[]
        s.Zeta_n=[]
        
        if not isinstance(s.eps_g1,np.ndarray):
            s.eps_g1=np.ones_like(s.pi_modes_init)*s.eps_g1
        
        s.boarders_to_switch_to_next_order= np.array([s.freq_minmax_to_Fit[0]
                                                   ,*np.diff(s.pi_modes_init)/2+s.pi_modes_init[:-1]
                                                   ,s.freq_minmax_to_Fit[-1]])
        
        s.mixed_modes_theo_n=[0]*s.num_orders
        s.Zeta_n=[0]*s.num_orders
        
        for i_np in range(s.num_orders):#[0:2]):
            s.mixed_modes_theo_n[i_np],s.Zeta_n[i_np],_=s.calc_theo_mixed_modes(pi_modes=s.pi_modes_init[i_np], q=s.q[i_np], dpi_Ms=s.dpi_Ms, dnu=s.dnu,eps_g1=s.eps_g1[i_np],freq_minmax=s.boarders_to_switch_to_next_order[i_np:i_np+2])
            
        s.mixed_modes_theo=np.concatenate(s.mixed_modes_theo_n)
        s.Zeta=np.concatenate(s.Zeta_n)
        
    def calc_theo_mixed_modes(s,pi_modes=False, gamma_modes=False, q=None, dpi_Ms=None, dnu=None,eps_g1=False,freq_minmax=False
                             , A_g_glitch=1, omega_g_glitch=1000, phi_g_glitch= 0.,Delta_g_glitch=0
                             , A_p_glitch=1, tau_p_glitch  =1, phi_p_glitch= 0.,b_p_glitch=0):
        
        if isinstance(pi_modes,bool):
            dpi_Ms=s.dpi_Ms
            dnu=s.dnu
            q=s.q
            
            if isinstance( freq_minmax,bool):
                freq_minmax=s.freq_minmax_to_Fit
            
            if hasattr(s,'gamma_modes_theo'):
                gamma_modes=s.gamma_modes_theo
            
            if s.buoyancy_glitch:
                A_g_glitch=s.A_g_glitch
                omega_g_glitch=s.omega_g_glitch
                phi_g_glitch= s.phi_g_glitch
                Delta_g_glitch=s.Delta_g_glitch
                
            if s.acoustic_glitch:
                A_p_glitch  = s.A_p_glitch
                tau_p_glitch= s.tau_p_glitch
                phi_p_glitch= s.phi_p_glitch
                b_p_glitch  = s.b_p_glitch
                
            else:   
                if not hasattr(s,'pi_modes_init'):
                    print('no pi_modes found; calculated with eps_p1_init')
                    s.pi_modes_theo = calc_pi_modes_theo(dnu=s.dnu,eps_p1=s.eps_p1_init,fminmax=freq_minmax
                                                        ,acoustic_glitch=s.acoustic_glitch_freq,A_p=s.A_p_glitch,tau_p=s.tau_p_glitch,phi_p=s.phi_p_glitch,b_p=s.b_p_glitch)
            pi_modes=s.pi_modes_theo
            
            if isinstance(q,np.ndarray) and not len(q)==len(s.pi_modes_theo):
                i=0
                while not np.all(np.abs(pi_modes[i:i+len(q)] - s.pi_modes_init)<1e-12):
                    i+=1
                q=np.pad(q, (i, len(pi_modes)-i-len(s.pi_modes_init)), mode="edge")
                
            if isinstance(s.eps_g1,np.ndarray) and not len(s.eps_g1)==len(pi_modes):
                print('eps_g1 and pi_modes not same length!! using mean value of eps_g1 for all orders')
                eps_g1=np.mean(s.eps_g1)
            else:
                eps_g1=s.eps_g1
                
        
        if isinstance(gamma_modes,bool):
            gamma_modes=calc_gamma_modes_theo(dpi_Ms=dpi_Ms,eps_g1=eps_g1,freq_minmax=freq_minmax,pi_modes=pi_modes,interp_kind=s.interp_kind
                                             ,buoyancy_glitch=s.buoyancy_glitch_freq,glitch_type=s.glitch_type,A_g=A_g_glitch,omega_g=omega_g_glitch,phi_g=phi_g_glitch,Delta_g=Delta_g_glitch)
            
        if not isinstance(pi_modes,(list,np.ndarray)):
            pi_modes=[pi_modes]
        
        mixed_modes_class=asympthotic_theory(pi_modes=pi_modes, gamma_modes=gamma_modes, q=q, dpi_Ms=dpi_Ms, dnu=dnu, interp_kind=s.interp_kind, glitch_type=s.glitch_type
                                                    ,buoyancy_glitch=s.buoyancy_glitch_phase, A_g=A_g_glitch, omega_g=omega_g_glitch, phi_g=phi_g_glitch, Delta_g=Delta_g_glitch
                                                    ,acoustic_glitch=s.acoustic_glitch_phase, A_p=A_p_glitch, tau_p  =  tau_p_glitch, phi_p=phi_p_glitch, b_p=b_p_glitch
                                                    )
        mixed_p_modes, mixed_g_modes = mixed_modes_class.couple_halley(maxiter=s.maxiter_halley)
        
        mixed_modes=np.sort(np.append(mixed_p_modes,mixed_g_modes))
        
        Zeta=mixed_modes_class.calc_ζp(mixed_modes)
        
        return mixed_modes, Zeta, gamma_modes
    
    
    #%%% template
        #%%%% make it
    def generate_template(s,pi_modes=False, eps_g1=False, q=False,freq_minmax=False
                           ,dpi_Ms=False,dnu=False,use_pre_fitted_l1_modes=False
                           ,A_g_glitch=False, omega_g_glitch=False, phi_g_glitch= False,Delta_g_glitch=False
                           ,A_p_glitch=False,   tau_p_glitch=False, phi_p_glitch= False,    b_p_glitch=False
                           ,SaveToClass=True,astropy_model=True,x=False,inflateFWHM=False
                           ,plot=True,ax=False,color=None):
        # freq_minmax=s.boarders_to_switch_to_next_order[i_np:i_np+2]
        
        if isinstance(freq_minmax, bool):
            freq_minmax=s.freq_minmax_to_Fit
            # freq_minmax=[-np.inf,np.inf]
            
        if not dnu:
            dnu=s.dnu
        if not dpi_Ms:
            dpi_Ms=s.dpi_Ms
        
        if SaveToClass:
            if s.discrete_orders:
                s.calc_theo_mixed_modes_discrete()
            else:
                s.mixed_modes_theo,s.Zeta,s.gamma_modes_theo=s.calc_theo_mixed_modes()
            mixed_modes_theo_n=s.mixed_modes_theo
            Zeta_n=s.Zeta
            
            if plot:
                if not ax:
                    plt.figure()
                    ax=plt.gca()
                    ax.plot(s.freq,s.psd+s.Bkg_func(s.freq),color='k',alpha=0.5,zorder=-10)
                ax.set_ylabel('PSD [ppm²/μHz]')
                ax.set_xlabel('ν (μHz)')
            
        else:
            mixed_modes_theo_n,Zeta_n,_=s.calc_theo_mixed_modes(pi_modes=pi_modes, gamma_modes=False, q=q, dpi_Ms=dpi_Ms, dnu=dnu,eps_g1=eps_g1
                                                               ,freq_minmax=freq_minmax
                                                               ,A_g_glitch=A_g_glitch,omega_g_glitch=omega_g_glitch,phi_g_glitch=phi_g_glitch,Delta_g_glitch=Delta_g_glitch
                                                               ,A_p_glitch=A_p_glitch,  tau_p_glitch=  tau_p_glitch,phi_p_glitch=phi_p_glitch,    b_p_glitch=    b_p_glitch)
        if not astropy_model:
            mask_n=(s.freq > freq_minmax[0]) & (s.freq < freq_minmax[1])
            x=s.freq[mask_n]
            
        if not hasattr(s,'freq_resolution'):
            # t_in_sec=s.time*(24*60*60)
            s.freq_resolution= 1/(s.T*(24*60*60))*1e6
        
        if not hasattr(s,'amplitude_interpolated_l0'):
            s.amplitude_interpolated_l0 = interp1d(s.modes0,s.modes0_lor['amp_fit']/s.y_to_scale,bounds_error=False,fill_value=(s.modes0_lor['amp_fit'][0]/s.y_to_scale,s.modes0_lor['amp_fit'][-1]/s.y_to_scale))
        
        if not hasattr(s,'fwhm_l0_func'):
            print('interpolated l=0 fwhm ')
            s.fwhm_l0_func = interp1d(s.modes0,s.modes0_lor['fwhm'],bounds_error=False,fill_value=(s.modes0_lor['fwhm'][0],s.modes0_lor['fwhm'][-1]))
        
        amplitude_interpolated_l0=s.amplitude_interpolated_l0
        
        if astropy_model:
            Bkg_model=custom_model(lambda x: s.Bkg_func(x)/s.y_to_scale)()
            template_scaled=Bkg_model
        else:
            y=s.Bkg_func(x)/s.y_to_scale
        
        if SaveToClass:
            modes1_lor_theo=Table(names=('amp_obs','amp_fit','x_0','fwhm','xmin','xmax','sig'))
        
        for l in [0,2,1]:
            if use_pre_fitted_l1_modes or l==0 or l==2:
                modesi_lor=getattr(s,f'modes{l}_lor')
                use_pre_fitted_lor=True
                modes_to_fit=modesi_lor#[(modesi_lor['x_0']>freq_minmax[0]-s.fwhm_l0_func(modesi_lor['x_0'])*3) &
                                       # (modesi_lor['x_0']<freq_minmax[1]+s.fwhm_l0_func(modesi_lor['x_0'])*3)]
                
            elif l==1:
                use_pre_fitted_lor=False
                modes_to_fit=mixed_modes_theo_n
                
                if SaveToClass:
                    freq_spacing_mixed_modes=np.zeros_like(modes_to_fit)
            
            if not astropy_model:
                lenmodes=len(modes_to_fit)
                x_0_i      =np.zeros(lenmodes)
                amplitude_i=np.zeros(lenmodes)
                fwhm_i     =np.zeros(lenmodes)
                
            for i_mode in range(len(modes_to_fit)):
                if use_pre_fitted_lor:
                    modesi_lor= modes_to_fit[i_mode]
                    modei     = modesi_lor['x_0']
                    xmin      = modesi_lor['xmin']
                    xmax      = modesi_lor['xmax']
                    fwhm      = modesi_lor['fwhm']
                    amplitude = modesi_lor['amp_fit']/s.y_to_scale
                    if s.fwhm_constrain_type and not l==1:
                        fwhm_bounds=(0.8*fwhm,1.2*fwhm)
                    else:
                        fwhm_bounds=(0.2*fwhm,3*fwhm)
                    amplitude_bounds=(1e-20,5*max(s.psd)/s.y_to_scale)
                    
                    # if plot:
                    #     if l==0:
                    #         ax.axvspan(xmin*0.995,xmax*1.002,alpha=0.3,color='limegreen')
                    #     if l==2:
                    #         ax.axvspan(xmin*0.995,xmax*1.002,alpha=0.3,color='orange')
                else:
                    modei=modes_to_fit[i_mode]
                    
                    fwhm_theo=s.fwhm_l0_func(modei)*(1-Zeta_n[i_mode]) #+ zeta*fwhm_g
                    if inflateFWHM:
                        fwhm_min= inflateFWHM*s.freq_resolution
                    else:
                        fwhm_min= 2*s.freq_resolution/np.pi
                    fwhm_convolved=np.sqrt(fwhm_theo**2+fwhm_min**2) 
                    fwhm=fwhm_convolved
                    amplitude   = amplitude_interpolated_l0(modei)*fwhm_theo/fwhm_convolved*1.5 #!!!
                    
                    if SaveToClass:
                        if i_mode==0:
                            freq_spacing_mixed_modes_i=np.mean(np.diff(modes_to_fit[0:2]))
                        elif i_mode==len(modes_to_fit)-1:
                            freq_spacing_mixed_modes_i=np.mean(np.diff(modes_to_fit[i_mode-1:]))
                        else:
                            freq_spacing_mixed_modes_i=np.mean(np.diff(modes_to_fit[i_mode-1:i_mode+2]))
                            
                        freq_spacing_mixed_modes[i_mode]=freq_spacing_mixed_modes_i
                        
                        xmin=modei-freq_spacing_mixed_modes_i/2
                        xmax=modei+freq_spacing_mixed_modes_i/2
                        
                        fwhm_bounds=(s.freq_resolution,s.dnu/3)
                        amplitude_bounds=(0.,5*max(s.psd)/s.y_to_scale)
                        
                        if plot: 
                            ax.axvline(modei,lw=0.5,color='lightblue')
                            ax.axvspan(xmin,xmax,alpha=0.3,color='lightblue')
                        
                        if any(abs(modei-s.modes0)<freq_spacing_mixed_modes_i/2+s.fwhm_l0_func(modei)) or any(abs(modei-s.modes2)<freq_spacing_mixed_modes_i/2+s.fwhm_l0_func(modei)): #!!!
                            if plot:
                                ax.axvline(modei,lw=0.5,color='red')
                            
                            fwhm_bounds=(s.freq_resolution,max(fwhm*1.3,s.freq_resolution*1.1)) #!!!
                            amplitude_bounds=(0.,amplitude*1.1)#!!!
                            # amplitude=0.
                            
                            # if any(abs(modei-s.modes0))<modei*l1_freq_tol or any(abs(modei-s.modes0))<modei*l1_freq_tol:
                            #     amplitude_bounds=(0.,1e-20)
                    
                        mask_lor=(s.freq > xmin) & (s.freq < xmax)
                        lorentz_tab=Table(rows=[[np.max(s.psd[mask_lor]),*[0.]*3,xmin,xmax,0.]]
                                         ,names=('amp_obs','amp_fit','x_0','fwhm','xmin','xmax','sig'))
                        
                        modes1_lor_theo=vstack([modes1_lor_theo, lorentz_tab])
                
                
                if astropy_model:
                    l_init_scaled = astropy_models.Lorentz1D( x_0=modei
                                                      ,amplitude=amplitude
                                                      ,fwhm=fwhm
                                                      ,bounds={"amplitude":amplitude_bounds
                                                               ,"x_0": (xmin,xmax)
                                                               ,"fwhm":fwhm_bounds
                                                               })
                    # if fix_fwhm:
                    #     l_init.fwhm.fixed = True
                    
                    template_scaled=template_scaled+l_init_scaled
                    
                else: 
                    amplitude_i[i_mode]= amplitude
                    x_0_i[i_mode]      = modei
                    fwhm_i     [i_mode]= fwhm
            
            if not astropy_model:
                y=y+lorentz_sum(x, x_0_i, amplitude_i, fwhm_i)
                
        if not astropy_model:
            return y
        
        if SaveToClass:
            if use_pre_fitted_l1_modes:
                s.freq_spacing_mixed_modes=freq_spacing_mixed_modes
            s.modes1_lor_theo=modes1_lor_theo
            s.template_scaled_init=template_scaled
            
            if plot:
                x_fit=np.linspace(s.freq[0],s.freq[-1],int(1e5))
                
                ax.plot(x_fit,template_scaled(x_fit)*s.y_to_scale,color=color)
                for pi_mode in s.pi_modes_init:
                    ax.axvline(pi_mode,color='blue',lw=2)        
            return template_scaled
        return template_scaled
        #%%%% fit it
    def fit_template_discrete(s, MCMC_or_dynesty=False):
       
        plt.figure()
        s.ax_templateFit=plt.gca()
        
        s.ax_templateFit.plot(s.freq,s.psd+s.Bkg_func(s.freq),color='k',alpha=0.5,zorder=-10)
        
        labels=["π-modes (s)", "ε_g", "q"]
        if s.fit_n_dependent_dpi:
            labels=labels+['ΔΠ']
        
        if MCMC_or_dynesty:
            s.pi_modes_init_uncertainties=np.zeros((s.num_orders,2))
            s.eps_g1_uncertainties       =np.zeros((s.num_orders,2))
            s.q_uncertainties            =np.zeros((s.num_orders,2))
            
            s.template_result         = [None]*s.num_orders
            s.template_result_sample  = [None]*s.num_orders
            s.template_result_weights = [None]*s.num_orders
        else:
            s.pi_modes_init_uncertainties=np.zeros(s.num_orders)
            s.eps_g1_uncertainties       =np.zeros(s.num_orders)
            s.q_uncertainties            =np.zeros(s.num_orders)
            
            if s.fit_n_dependent_dpi:
                s.dpi_n_dependent_uncertainties  =np.zeros(s.num_orders)
                s.dpi_n_dependent = np.zeros(s.num_orders)
        
        for i_np in range(s.num_orders):
            
            p0=[s.pi_modes_init[i_np], s.eps_g1[i_np], s.q[i_np]]
            pi_minmax   = [s.pi_modes_init[i_np] - 0.3 * s.dnu
                         , s.pi_modes_init[i_np] + 0.3 * s.dnu]
            
            eps_g1_minmax=[s.eps_g1[i_np]-0.5,s.eps_g1[i_np]+0.5]
            
            bounds=[(pi_minmax[0],pi_minmax[1]),(eps_g1_minmax[0],eps_g1_minmax[1]),(0.01,1.)]
            
            if s.fit_n_dependent_dpi:
                p0=p0+[s.dpi_Ms]
                bounds=bounds+[(s.dpi_Ms*0.9,s.dpi_Ms*1.1)]
            
            mask_psdn=(s.boarders_to_switch_to_next_order[i_np]<=s.freq) & (s.freq<s.boarders_to_switch_to_next_order[i_np+1])
            xn=s.freq[mask_psdn]
            yn=(s.psd[mask_psdn]+s.Bkg_func(xn))/s.y_to_scale
            
            x_fit=np.linspace(*s.boarders_to_switch_to_next_order[i_np:i_np+2],10000)
            
            def _to_minimize(params):
                
                if s.fit_n_dependent_dpi:
                    dpi_n=params[-1]
                    params=params[:-1]
                else:
                    dpi_n=s.dpi_Ms
                    
                pi_mode_n,eps_g1_n,q_n=params
                    
                if MCMC_or_dynesty=='MCMC':
                    log_prior=0
                    if not (bounds[0][0] <= pi_mode_n <= bounds[0][1]):
                        log_prior= -np.inf
                    if not (bounds[1][0] <= eps_g1_n <= bounds[1][1]):
                        log_prior= -np.inf
                    if not (bounds[2][0] <= q_n <= bounds[2][1]):
                        log_prior= -np.inf
                        
                    if not np.isfinite(log_prior):
                        return -np.inf
                    
                eps_g1_n = eps_g1_n % 1
                
                y_fit=s.generate_template(pi_modes=pi_mode_n, eps_g1=eps_g1_n, q=q_n,dpi_Ms=dpi_n,freq_minmax=s.boarders_to_switch_to_next_order[i_np:i_np+2]
                                          # ,A_g=A_g_glitch,omega_g=omega_g_glitch,phi_g=phi_g_glitch,Delta_g=Delta_g_glitch
                                          ,SaveToClass=False,plot=False,astropy_model=False,inflateFWHM=s.inflateFWHM)
                
                #return neg_log_likelihood_X22dof(x=xn, y=yn,y_fit=y_fit) * (int(dynestyFit)*(-2)+1)
                if MCMC_or_dynesty=='dynesty':
                    return           -neg_log_likelihood_X22dof(x=xn, y=yn,y_fit=y_fit) # / len(y_fit)
                if MCMC_or_dynesty=='MCMC':
                    return log_prior -neg_log_likelihood_X22dof(x=xn, y=yn,y_fit=y_fit) 
                else:
                    return neg_log_likelihood_X22dof(x=xn, y=yn,y_fit=y_fit)  
            
            
            if MCMC_or_dynesty=='dynesty':
                def prior_transform(u):
                    v = np.zeros(3)
                    for i in range(3):
                        v[i]=bounds[i][0] + u[i] * (bounds[i][1] - bounds[i][0])
                    return v
                
                sampler = dynesty.NestedSampler(_to_minimize, prior_transform, ndim=3)
                sampler.run_nested(dlogz=0.1)
            
                res = sampler.results
            
                s.template_result_sample[i_np]  = res.samples
                s.template_result_weights[i_np] = np.exp(res.logwt - res.logz[-1])
            
                statistics_type='weighted probability density'
                
            elif MCMC_or_dynesty=='MCMC':
                ndim = 3
                nwalkers = 32
                # pos = p0 + 0.01*np.random.randn(nwalkers, ndim)
                pos = np.array([np.random.uniform(low=b[0], high=b[1], size=nwalkers) for b in bounds]).T
                
                nsteps=5000
                thin=10
                sampler = emcee.EnsembleSampler(nwalkers, ndim, _to_minimize)
                sampler.run_mcmc(pos, nsteps, progress=True)
                s.template_result[i_np] = sampler
                s.template_result_sample[i_np] = sampler.get_chain(discard=int(nsteps*0.01),flat=True,thin=thin) #!!!
                s.template_result_weights[i_np]=None#np.ones_like(s.template_result_sample[i_np])#None
                
                # tau = sampler.get_autocorr_time()
                # print(50*tau)
                
                statistics_type='probability density'
                
            if MCMC_or_dynesty:
                # corner.corner(s.template_result_sample[i_np],weights=s.template_result_weights[i_np],labels=labels
                #              ,quantiles=[0.16, 0.5, 0.84],show_titles=True,title_fmt=".3f")
                
                try:
                    results_n,uncertainties_n=plot_cornerplot(s.template_result_sample[i_np].T,labels=labels,xi=None,L=s.template_result_weights[i_np],labelcbar='',mask=None, levels=[0.68, 0.95]
                                                             ,statistics_type=statistics_type,bins = 50,num_models_goodxi=25,median_or_mode='mode'
                                                             ,kde_2D_plot=False,idx_for_circular_parameters=[1],plot=True,thin=20)
                    pi_mode_n,eps_g1_n,q_n=results_n
                    pi_mode_n_uncertainties,eps_g1_n_uncertainties,q_n_uncertainties=uncertainties_n
                except:
                    print('something went wrong')
                    # result_n=np.sum(s.template_result_sample[i_np] * s.template_result_weights[i_np][:, None], axis=0)
                    
                    
                    pi_mode_n,eps_g1_n,q_n=results_n
                    pi_mode_n_uncertainties,eps_g1_n_uncertainties,q_n_uncertainties=np.ones_like(results_n)
                
                s.pi_modes_init_uncertainties[i_np,:]=pi_mode_n_uncertainties
                s.eps_g1_uncertainties[i_np,:]       =eps_g1_n_uncertainties
                s.q_uncertainties[i_np,:]            =q_n_uncertainties
                dpi_n=s.dpi_Ms

            else:
                if s.minimizer=='minimize':
                    result_n=minimize(_to_minimize,p0,method='L-BFGS-B',bounds=bounds)
                    
                elif s.minimizer=='differential_evolution':
                    
                    result_n=differential_evolution(_to_minimize,bounds=bounds,polish=False)
                    result_n=minimize(_to_minimize,result_n.x,method='L-BFGS-B',bounds=bounds)#,options={"ftol": 1e-14,"gtol": 1e-10,"maxiter": 5000})
                    # result_n=minimize(_to_minimize,result_n.x,method='BFGS')
                    
                s.result_minimize= result_n
                cov = result_n.hess_inv.todense()
                errors = np.sqrt(np.diag(cov))
                try:
                    
                    H = nd.Hessian(_to_minimize)(result_n.x)
                    
                    cov = np.linalg.inv(H)
                    errors = np.sqrt(np.diag(cov)) 
                except:
                    print('num Hessian failed')
                    
                if s.inflateFWHM:
                    smoothingFactor=s.inflateFWHM
                else:
                    smoothingFactor=1
                errors=errors* np.sqrt(-result_n.fun/9 /smoothingFactor) 
                
                if s.fit_n_dependent_dpi:
                    s.pi_modes_init_uncertainties[i_np],s.eps_g1_uncertainties[i_np],s.q_uncertainties[i_np],s.dpi_n_dependent_uncertainties[i_np]=errors
                    pi_mode_n,eps_g1_n,q_n,dpi_n=result_n.x
                    s.dpi_n_dependent[i_np]=dpi_n
                else:
                    s.pi_modes_init_uncertainties[i_np],s.eps_g1_uncertainties[i_np],s.q_uncertainties[i_np]=errors
                    pi_mode_n,eps_g1_n,q_n=result_n.x
                    dpi_n=s.dpi_Ms
            
            s.pi_modes_init[i_np]=pi_mode_n
            s.eps_g1[i_np]       =eps_g1_n%1
            s.q[i_np]            =q_n
                
            template_scaled=s.generate_template(pi_modes=pi_mode_n, eps_g1=eps_g1_n, q=q_n,dpi_Ms=dpi_n,freq_minmax=s.boarders_to_switch_to_next_order[i_np:i_np+2],SaveToClass=False,plot=False,astropy_model=True)
            template_scaled_smoothed=s.generate_template(pi_modes=pi_mode_n, eps_g1=eps_g1_n, q=q_n,dpi_Ms=dpi_n,freq_minmax=s.boarders_to_switch_to_next_order[i_np:i_np+2],SaveToClass=False,plot=False,astropy_model=True,inflateFWHM=s.inflateFWHM)
            
            s.ax_templateFit.plot(x_fit,template_scaled(x_fit)*s.y_to_scale)
            s.ax_templateFit.plot(x_fit,template_scaled_smoothed(x_fit)*s.y_to_scale,color='red')
            print(i_np+1,'/',s.num_orders,'  done')
        
        s.eps_p1_init  =s.pi_modes_init % s.dnu / s.dnu
        s.eps_p1       =s.eps_p1_init
        s.pi_modes     =s.pi_modes_init
        
        ng  = (1/(s.pi_modes_init*1e-6)) // s.dpi
        s.mean_gamma_modes_sec=(s.eps_g1+ng) * s.dpi
        
        
    def fit_template_all(s,scale_free_parameters=True,SaveToClass=True,plot=True):
        
        if plot:
            plt.figure()
            s.ax_templateFit=plt.gca()
        
            s.ax_templateFit.plot(s.freq,s.psd+s.Bkg_func(s.freq),color='grey',alpha=0.5,zorder=-10)
            # s.ax_templateFit.plot(s.freq,psd_noisy+s.Bkg_func(s.freq),color='k',alpha=0.5,zorder=-10)
        
        if s.MCMC_or_dynesty:
            scale_free_parameters=False
        
        q_minmax=[]
        
        labels=[*["q"]*s.num_orders]
        p0    =[*s.q]
        
        if s.acoustic_glitch:
            p0+=[s.eps_p1_init]
            pi_minmax   = [(s.eps_p1_init-0.2,s.eps_p1_init+0.2)]
            labels+=["ε_p"]
        else:
            p0+=[*s.pi_modes_init]
            pi_minmax=[]
            labels+=[*["π-modes"]*s.num_orders]
            
        if s.buoyancy_glitch:
            p0+=[s.eps_g1]
            labels+=["ε_g"]
            eps_g1_minmax=[(s.eps_g1-0.5,s.eps_g1+0.5)]
        else:
            p0+=[*s.eps_g1]
            labels+=[*["ε_g"]*s.num_orders]
            eps_g1_minmax=[]
            
        for i_np in range(s.num_orders):
            q_minmax+=[(0.01,0.9)]
            if not s.acoustic_glitch:
                pi_minmax   += [(s.pi_modes_init[i_np] - 0.25 * s.dnu
                                ,s.pi_modes_init[i_np] + 0.25 * s.dnu)]
            if not s.buoyancy_glitch:
                eps_g1_minmax+=[(s.eps_g1[i_np]-0.5,s.eps_g1[i_np]+0.5)]
        
        bounds=q_minmax+pi_minmax+eps_g1_minmax
        
        if s.acoustic_glitch:
            
            labels+=["A_p","τ_p","φ_p","b_p"]
            p0+= [s.A_p_glitch,s.tau_p_glitch,s.phi_p_glitch,s.b_p_glitch] 
            
            A_p_glitch_minmax    = (0,5) 
            tau_p_glitch_minmax  = (0,0.05)
            phi_p_glitch_minmax  = (s.phi_p_glitch-np.pi,s.phi_p_glitch+np.pi)
            b_p_glitch_minmax    = (0,0.005)
            
            bounds+=[A_p_glitch_minmax,tau_p_glitch_minmax,phi_p_glitch_minmax,b_p_glitch_minmax]
        
        if s.buoyancy_glitch:
            labels+=["A_g","ω_g","φ_g"]
            p0+= [s.A_g_glitch,s.omega_g_glitch,s.phi_g_glitch] 
            Delta_g_glitch_minmax = []
            if s.glitch_type=='gaussian':
                if not s.fixed_delta_g_glitch:
                    p0+=[0.]
                    labels+=["Δ"]
                    Delta_g_glitch_minmax = [(0,1000)]
                A_g_glitch_minmax       = (max([0,s.A_g_glitch*0+0.1]) ,min([3000 ,max([s.A_g_glitch*2  ,1000.])]))
            else:
                A_g_glitch_minmax       = (max([0,s.A_g_glitch*0+0.1]) ,min([100  ,max([s.A_g_glitch*2    ,5.])]))
                

            omega_g_glitch_minmax   = (max([1,s.omega_g_glitch*0+1]),min([5000,max([s.omega_g_glitch*3,2000])]))
            phi_g_glitch_minmax     = (s.phi_g_glitch-np.pi,s.phi_g_glitch+np.pi)
            
            bounds+=[A_g_glitch_minmax,omega_g_glitch_minmax,phi_g_glitch_minmax]+Delta_g_glitch_minmax
            
        mask=(s.freq>s.freq_minmax_to_Fit[0]) & (s.freq<s.freq_minmax_to_Fit[1])
        x_freq=s.freq[mask]
        y_psd =(s.psd[mask]+s.Bkg_func(x_freq))/s.y_to_scale
        
        if s.fit_dpi:
            p0+= [s.dpi_Ms]
            labels+=['ΔΠ']
            bounds+=[(s.dpi_Ms*0.95,s.dpi_Ms*1.05)] 
            
        x_fit=np.linspace(s.freq_minmax_to_Fit[0],s.freq_minmax_to_Fit[1],10000)
        s.cornerplot_labels=labels
        mins = np.array([b[0] for b in bounds])
        maxs = np.array([b[1] for b in bounds])
        if scale_free_parameters:
            def normalize(p0):
                return (p0 - mins) / (maxs - mins)
            
            p0_normalized=normalize(p0)
                
            def denormalize(params):
                return mins + params * (maxs - mins)
            
            bounds_normalized=[(0., 1.)] * len(p0)
            
        else:
            p0_normalized=p0
            bounds_normalized=bounds

        s.addNoise=False
        s.calc_hessian_with_bounds=False

        def _to_minimize(params):
            if scale_free_parameters:
                params=denormalize(params)
                
            if s.MCMC_or_dynesty=='MCMC' or s.calc_hessian_with_bounds:
                if np.any(params < mins) or np.any(params > maxs):
                    if s.calc_hessian_with_bounds:
                        return 1e20
                    return -np.inf
            
            q            = params[:s.num_orders]
            params       = params[s.num_orders:] 
            
            if not s.acoustic_glitch:
                pi_modes = params[:s.num_orders]
                params   = params[s.num_orders:] 
            else:
                eps_p1   = params[0]
                params   = params[1:] 
                
                if not s.MCMC_or_dynesty=='dynesty':
                    eps_p1 = eps_p1 % 1
                
            if not s.buoyancy_glitch:
                eps_g1   = params[:s.num_orders]
                params   = params[s.num_orders:] 
            else:
                eps_g1   = params[0]
                params   = params[1:] 
            
            if s.fit_dpi:
                dpi_Ms   = params[-1]
                params   = params[:-1]
            else:
                dpi_Ms=s.dpi_Ms
                
            if s.acoustic_glitch:
                A_p_glitch,tau_p_glitch,phi_p_glitch, b_p_glitch=params[:4]
                
                omega_mid=2*np.pi*np.mean(s.freq_minmax_to_Fit)
                A_eff = A_p_glitch * np.exp(-2 * b_p_glitch**2 * omega_mid**2)
                if A_eff>0.5:
                    if s.MCMC_or_dynesty:
                        return -np.inf
                    else:
                        return 1e20
                
                if not s.MCMC_or_dynesty=='dynesty':
                    phi_p_glitch=((phi_p_glitch+np.pi)%(2*np.pi))-np.pi
            else:
                A_p_glitch=False
                tau_p_glitch=False
                phi_p_glitch=False
                b_p_glitch=False
                
            if s.buoyancy_glitch:
                if s.glitch_type=='step-like' or s.fixed_delta_g_glitch:
                    A_g_glitch,omega_g_glitch,phi_g_glitch=params[-3:]
                    Delta_g_glitch=0
                else:
                    A_g_glitch,omega_g_glitch,phi_g_glitch,Delta_g_glitch=params[-4:]
                    
                if not s.MCMC_or_dynesty=='dynesty':
                    phi_g_glitch=((phi_g_glitch+np.pi)%(2*np.pi))-np.pi
            else:
                A_g_glitch=False
                omega_g_glitch=False
                phi_g_glitch=False
                Delta_g_glitch=False
        
            if not s.MCMC_or_dynesty=='dynesty':
                eps_g1 = eps_g1 % 1
            
            if s.acoustic_glitch:
                pi_modes = calc_pi_modes_theo(s.dnu, eps_p1, s.freq_minmax_to_Fit)
                
                pi_modes_glitch = calc_pi_modes_theo(dnu=s.dnu,eps_p1=eps_p1,fminmax=s.freq_minmax_to_Fit
                                             ,acoustic_glitch=s.acoustic_glitch,A_p=A_p_glitch,tau_p=tau_p_glitch,phi_p=phi_p_glitch,b_p=b_p_glitch)
                eps_p1=pi_modes_glitch%s.dnu/s.dnu
                if np.any(eps_p1<pi_minmax[0][0]) or np.any(eps_p1>pi_minmax[0][1]):
                    if s.MCMC_or_dynesty:
                        return -np.inf
                    else:
                        return 1e20
            
            y_fit=s.generate_template(pi_modes=pi_modes, eps_g1=eps_g1, q=q, dpi_Ms=dpi_Ms,freq_minmax=s.freq_minmax_to_Fit
                                      ,A_g_glitch=A_g_glitch,omega_g_glitch=omega_g_glitch,phi_g_glitch=phi_g_glitch,Delta_g_glitch=Delta_g_glitch
                                      ,A_p_glitch=A_p_glitch,  tau_p_glitch=  tau_p_glitch,phi_p_glitch=phi_p_glitch,    b_p_glitch=    b_p_glitch
                                      ,SaveToClass=False,plot=False,astropy_model=False,inflateFWHM=s.inflateFWHM)
            
            if s.addNoise:
                y_psd_to_fit = s.y_psd_noisy
            else:
                y_psd_to_fit = y_psd
            
            if s.MCMC_or_dynesty:
                return -neg_log_likelihood_X22dof(x=x_freq, y=y_psd_to_fit, y_fit=y_fit) 
            else:
                return  neg_log_likelihood_X22dof(x=x_freq, y=y_psd_to_fit, y_fit=y_fit)  
        
        minimizer_options = {"method": "L-BFGS-B"
                            ,"options": {"ftol": 1e-14, "gtol": 1e-12, "maxiter": 5000}}
        
        s.idx_for_circular_parameters=np.where(np.isin(labels, ['ε_g','φ_p','φ_g']))[0]
        ndim = len(p0)
        if s.MCMC_or_dynesty:
            if s.MCMC_or_dynesty=='MCMC':
                nwalkers = ndim*10#32
                # pos = p0 + 0.01*np.random.randn(nwalkers, ndim)
                pos = np.array([np.random.uniform(low=b[0], high=b[1], size=nwalkers) for b in bounds_normalized]).T
        
                nsteps=5000
                thin=10
                sampler = emcee.EnsembleSampler(nwalkers, ndim, _to_minimize)
                sampler.run_mcmc(pos, nsteps, progress=True)
                s.template_result = sampler
                s.template_result_sample = sampler.get_chain(discard=int(100),flat=True,thin=thin) 
                s.template_result_weights=None#np.ones_like(s.template_result_sample[i_np])#None
                
                s.statistics_type='probability density'
                
            elif s.MCMC_or_dynesty=='dynesty':
                def prior_transform(u):
                    if scale_free_parameters:
                        return u
                    v = np.zeros(ndim)
                    for i in range(ndim):
                        low, high = bounds[i]
                        v[i] = low + u[i] * (high - low)
                    return v
                
                thin=10
                nlive=2000 #!!!
                sampler = dynesty.DynamicNestedSampler(_to_minimize, prior_transform, ndim=ndim,nlive = nlive
                                                       ,periodic=s.idx_for_circular_parameters)
                sampler.run_nested()#dlogz=0.1) #!!!
            
                res = sampler.results
                
                s.template_result         = sampler
                s.template_result_sample  = res.samples
                s.template_result_weights = np.exp(res.logwt - res.logz[-1])
            
                s.statistics_type='weighted probability density'
                
            
            # corner.corner(s.template_result_sample,weights=s.template_result_weights,labels=labels
            #              ,quantiles=[0.16, 0.5, 0.84],show_titles=True,title_fmt=".3f")
            
            try:
                results,uncertainties=s.plot_fit_template_corner(plot=False,thin=thin)
            except:
                s.save(filename=s.starname+'_failed.npz')
                return
            
            s.result_MCMC=results
            s.result_MCMC_errors=uncertainties

            inflate = 1.0
            bounds_mcmc = [(result - inflate * sig[0],result + inflate * sig[1]) for result, sig in zip(results, uncertainties)]
            
            MCMC_or_dynesty=s.MCMC_or_dynesty
            s.MCMC_or_dynesty=False
            result_minimize=minimize(_to_minimize,results,method=minimizer_options['method'],bounds= bounds_mcmc,options=minimizer_options['options'])
            s.MCMC_or_dynesty=MCMC_or_dynesty
            
            s.result_minimize=result_minimize.x
            s.result_uncertainties= s.result_MCMC_errors
        
        else:
            if s.iterations_for_fit:
                niterations_for_fit=s.iterations_for_fit+1
            else:
                niterations_for_fit=1
                
            s.results_template_fit_with_noise=[0]*(niterations_for_fit-1)
            s.results_template_fit_with_noise_only_de=[0]*niterations_for_fit
            
            
            for n_iter in range( niterations_for_fit):

                # psd_noisy= s.psd * np.random.chisquare(df=2, size=s.psd.shape) / 2
                alpha = 1
                noise = np.random.chisquare(df=2, size=x_freq.shape)
                psd_noisy_factor = (1 + alpha*(noise - 1))
                
                s.y_psd_noisy=(s.psd[mask]+s.Bkg_func(x_freq)*psd_noisy_factor)/s.y_to_scale
                
                if s.minimizer=='minimize':
                    result_n=minimize(_to_minimize,p0_normalized,method=minimizer_options['method'],bounds=bounds_normalized,options=minimizer_options['options'])
                    
                elif s.minimizer=='differential_evolution':
                    print("start ",n_iter)
                    result_n=differential_evolution(_to_minimize,bounds=bounds_normalized,polish=False,tol=1e-2,popsize=200)
                    s.results_template_fit_with_noise_only_de[n_iter]= denormalize(result_n.x) 
                    # print(denormalize(result_n.x))
                    print('polish')
                    result_n=minimize(_to_minimize,result_n.x,method=minimizer_options['method'],bounds=bounds_normalized
                                      ,options=minimizer_options['options']
                                      )
                    print("finished ",n_iter)
        
                elif s.minimizer == 'basinhopping':
                    result_n = basinhopping(_to_minimize, p0_normalized,  minimizer_kwargs=minimizer_options
                                            ,niter=10000)
                if n_iter==0:
                    s.addNoise=True
                    if scale_free_parameters:
                        s.result_minimize= denormalize(result_n.x)
                    else:
                        s.result_minimize=result_n.x
                        
                    cov = result_n.hess_inv.todense()
                    s.errors_auto = np.sqrt(np.diag(cov))
                    
                    print('hessian')
                    try:
                        s.calc_hessian_with_bounds=True
                        H = nd.Hessian(_to_minimize)(result_n.x)
                        
                        cov = np.linalg.inv(H)
                        s.errors = np.sqrt(np.diag(cov)) 
                    except:
                        print('num Hessian failed')
                    
                else:
                    if scale_free_parameters:
                        s.results_template_fit_with_noise[n_iter-1]= denormalize(result_n.x)
                    else:
                        s.results_template_fit_with_noise[n_iter-1]=result_n.x
        
        if SaveToClass:
            s.q                = s.result_minimize[            :s.num_orders]
            result_n_temp      = s.result_minimize[s.num_orders:]
            
            s.q_uncertainties                = s.result_uncertainties[            :s.num_orders]
            result_n_temp_uncertainties      = np.array(s.result_uncertainties)[s.num_orders:]
            
            if not s.acoustic_glitch:
                s.pi_modes_init= result_n_temp[:s.num_orders]
                s.eps_p1_init  = s.pi_modes_init % s.dnu / s.dnu
                result_n_temp  = result_n_temp[s.num_orders:]
                
                s.pi_modes_uncertainties     = result_n_temp_uncertainties[:s.num_orders]
                s.eps_p1_init_uncertainties  = np.array(s.pi_modes_uncertainties) / s.dnu
                result_n_temp_uncertainties  = result_n_temp_uncertainties[s.num_orders:]
            else:
                s.eps_p1_init  = result_n_temp[0]%1
                result_n_temp  = result_n_temp[1:]
                
                s.eps_p1_init_uncertainties  = result_n_temp_uncertainties[0]
                result_n_temp_uncertainties  = result_n_temp_uncertainties[1:]
            
            
            if not s.buoyancy_glitch:
                s.eps_g1       = result_n_temp[:s.num_orders]%1
                result_n_temp  = result_n_temp[s.num_orders:]
                
                s.eps_g1_uncertainties       = result_n_temp_uncertainties[:s.num_orders]
                result_n_temp_uncertainties  = result_n_temp_uncertainties[s.num_orders:]
            else:
                s.eps_g1       = result_n_temp[0]%1
                result_n_temp  = result_n_temp[1:]
                
                s.eps_g1_uncertainties       = result_n_temp_uncertainties[0]
                result_n_temp_uncertainties  = result_n_temp_uncertainties[1:]
                
            if s.fit_dpi:
                s.dpi          = result_n_temp[-1]*1e6
                result_n_temp  = result_n_temp[:-1]
                
                s.dpi_uncertainties          = result_n_temp_uncertainties[-1]*1e6
                result_n_temp_uncertainties  = result_n_temp_uncertainties[:-1]
                
            if s.acoustic_glitch:
                s.A_p_glitch,s.tau_p_glitch,s.phi_p_glitch,s.b_p_glitch = result_n_temp[:4]
                s.phi_p_glitch=((s.phi_p_glitch+np.pi)%(2*np.pi))-np.pi
                
                s.A_p_glitch_uncertainties,s.tau_p_glitch_uncertainties,s.phi_p_glitch_uncertainties,s.b_p_glitch_uncertainties = result_n_temp_uncertainties[:4]
                s.phi_p_glitch=((s.phi_p_glitch+np.pi)%(2*np.pi))-np.pi
                
            if s.buoyancy_glitch:
                if s.glitch_type=='gaussian' and not s.fixed_delta_g_glitch:
                    s.A_g_glitch,s.omega_g_glitch,s.phi_g_glitch,s.Delta_g_glitch = result_n_temp[-4:]
                    
                    s.A_g_glitch_uncertainties,s.omega_g_glitch_uncertainties,s.phi_g_glitch_uncertainties,s.Delta_g_glitch_uncertainties = result_n_temp_uncertainties[-4:]
                else:
                    s.A_g_glitch,s.omega_g_glitch,s.phi_g_glitch = result_n_temp[-3:]
                    
                    s.A_g_glitch_uncertainties,s.omega_g_glitch_uncertainties,s.phi_g_glitch_uncertainties = result_n_temp_uncertainties[-3:]
                    
                s.phi_g_glitch=((s.phi_g_glitch+np.pi)%(2*np.pi))-np.pi
        
        if s.inflateFWHM:
            smoothingFactor=s.inflateFWHM
        else:
            smoothingFactor=1
        
        s.eps_p1       =s.eps_p1_init #!!!
        s.pi_modes     =s.pi_modes_init
        
        ng  = (1/(s.pi_modes_init*1e-6)) // s.dpi
        s.mean_gamma_modes_sec=(s.eps_g1+ng) * s.dpi
        
        template_scaled=s.generate_template(freq_minmax=s.freq_minmax_to_Fit,SaveToClass=True)
        if plot:
            template_scaled_smoothed=s.generate_template(inflateFWHM=s.inflateFWHM,freq_minmax=s.freq_minmax_to_Fit,SaveToClass=False)
        
            s.ax_templateFit.plot(x_fit,template_scaled(x_fit)*s.y_to_scale)
            s.ax_templateFit.plot(x_fit,template_scaled_smoothed(x_fit)*s.y_to_scale,color='red')
        
    
    #%%% plotters
    def plot_lc(s):
        plot_lightcurve(time=s.time,flux=s.flux,axlc=False,JD=s.JD
                           ,color="k",marker='x',label='',ylabel="Flux",xlabel=False
                           ,ls='None',ms=2,zorder=None
                            )
        
    def plot_template(s,event):
        plt.figure(figsize=(10,5))
        ax=plt.gca()
        # ax.set_xlim(33,57)
        # ax.set_ylim(400,40000)
        template_scaled=s.generate_template(ax=ax,inflateFWHM=s.inflateFWHM)
        
        ax.plot(s.freq,s.psd+s.Bkg_func(s.freq),color='k',alpha=0.2,zorder=-10)
        ax.plot(s.freq_sm,s.psd_sm+s.Bkg_func(s.freq_sm),color='k',alpha=1,zorder=-10)
        
        if s.discrete_orders:
            for i_np in range(s.num_orders):
                x_fit=np.linspace(*s.boarders_to_switch_to_next_order[i_np:i_np+2],10000)
                template_scaled=s.generate_template(pi_modes=s.pi_modes_init[i_np], eps_g1=s.eps_g1[i_np], q=s.q[i_np],freq_minmax=s.boarders_to_switch_to_next_order[i_np:i_np+2],SaveToClass=False)
                ax.plot(x_fit,template_scaled(x_fit)*s.y_to_scale)
                
        ## poster 
        if False:
            darkblue=(21,23,65)
            orange=(250,135,48)
            lightblue=(116, 184, 213)
            blue= (70, 131, 249)
            
            darkblue=RGB_to_hex(darkblue)
            orange=RGB_to_hex(orange)
            lightblue=RGB_to_hex(lightblue)
            blue=RGB_to_hex(blue)
            
            plt.figure(figsize=(15,5))
            ax=plt.gca()
            # ax.set_xlim(33,57)
            # ax.set_ylim(400,40000)
            template=s.generate_template(inflateFWHM=s.inflateFWHM,plot=False)
    
            ax.plot(s.freq,s.psd+s.Bkg_func(s.freq),color=darkblue,alpha=1,zorder=-10,label='PSD')
            ax.plot(s.freq_sm,s.psd_sm+s.Bkg_func(s.freq_sm),color=orange,alpha=1,zorder=-10,label='smoothed PSD',lw=2)
        
            x_fit=np.linspace(s.freq[0],s.freq[-1],int(1e5))
            ax.plot(x_fit,template_scaled(x_fit)*s.y_to_scale,color='w',label='template fit')
            
            # ax.legend(frameon=False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            ax.set_xlabel('Frequency (μHz)')
    
    def fit_q(s):

        if not hasattr(s, 'q_uncertainties'):
            s.q_uncertainties=np.ones((len(s.q),2))*1e-20
        def calc_q(modes,fL,fD):
            cyc2ang=(2*np.pi)*1e-6
            dnu   = s.dnu*cyc2ang
            omega = modes*cyc2ang
            
            q= 1/(dnu*s.dpi_Ms) *(np.pi*(fL+omega**2*fD)/(2*omega**2))
            return q
        if isinstance(s.q_uncertainties[0],(list,np.ndarray)):
            sigma=np.mean(s.q_uncertainties,axis=1)
        else:
            sigma=s.q_uncertainties
            
        fitted_params,_=curve_fit(calc_q, s.pi_modes, s.q
                                 ,sigma=sigma
                                 ,bounds=([0, 0], [np.inf, np.inf]))
        
        freq=np.linspace(min(s.pi_modes),max(s.pi_modes),1000)
        q_fitted=calc_q(freq,*fitted_params)
        
        plt.figure(figsize=(6,3.5))
        ax=plt.gca()
        ax.errorbar(s.pi_modes/s.dnu,s.q,yerr=np.array(s.q_uncertainties).T,marker='o',ls='None',color='k')
        # ax.plot(freq/s.dnu,q_fitted,'r')
        ax.set_xlabel('ν/Δν')
        ax.set_ylabel('q')
        
    def plot_dpi_obs(s):
         
        plt.figure(figsize=(8,5))
        ax=plt.gca()
       
        dpi_obs=np.abs(np.diff(1/s.modes1*1e6))
        means=[(s.modes1[i] + s.modes1[i+1]) / 2 for i in range(len(s.modes1)-1)]
        ax.plot(means,dpi_obs,'o',color='dodgerblue')
        ax.axhline(s.dpi,color='r')
        ax.set_xlabel('ν (μHz)')
        ax.set_ylabel('ΔΠ (s)',color='dodgerblue')
        
    
        ax.plot(s.mixed_modes_theo,s.Zeta*s.dpi,'kx',mew=2)
        
        freq=np.linspace(s.mixed_modes_theo[0],s.mixed_modes_theo[-1],1000)
        zeta=asympthotic_theory(pi_modes=s.pi_modes, q=s.q, dpi_Ms=s.dpi_Ms,dnu= s.dnu, interp_kind=s.interp_kind).calc_ζp(freq)
        ax.plot(freq,zeta*s.dpi,'k',zorder=-2)
        
        ax.set_xlabel('ν (μHz)')
        
        
        ax2=ax.twinx()
        ax2.set_ylim(min(zeta)*0.97,1.01)
        ax2.set_ylabel('ζ$_p$',color='k')
        ax.set_ylim(min(zeta)*s.dpi*0.97,s.dpi*1.01)
        
    def plot_fit_template_corner(s,plot=True,bins=30,thin=1):
        sample=s.template_result_sample.T[:,::thin]
        weights=s.template_result_weights[::thin]
        results,uncertainties=plot_cornerplot(sample,labels=s.cornerplot_labels,xi=None,L=weights,labelcbar='',mask=None, levels=[0.68, 0.95]
                       ,statistics_type=s.statistics_type,bins = bins,median_or_mode='mode'
                       ,kde_2D_plot=False,idx_for_circular_parameters=[s.idx_for_circular_parameters[0]],plot=plot) #!!!
            
        return results,uncertainties
    
#%% Echelle       
class Echelle:
    #%%% echelle 
    def __init__(es,s,ax_echelle=False,vMinMax=True
                 
                 ,sqrt_psd=False,double=False,xaxis_offset=0.
                 
                 ,colorbar=False,cmap="Greys"
                 ,lcolor=None,color_theo_modes='red',color_pigmodes=['cyan','violet'],legend=False
                 ,amp_as_markersize=True,minmarkersize=20,markersize_pigmodes=50,resolution=200
                 
                 ,add_streched=False,dpi=100.,q=0.1,eps_p1=0.5,eps_g1=0.5
                 ,frequency_dependent_eps_q=False
                 ,estimate_eps_using_modes1=False
                 ,plot_streched_theo_modes=False
                 
                 ,paper_format=False
                 ):
        
        es.s = s
        s.es = es
        
        es.estimate_eps_using_modes1=estimate_eps_using_modes1
        
        es.resolution=resolution
        
        s.xaxis_offset=xaxis_offset
        es.cmap=cmap
        es.double=double
        if not es.double:
            s.xaxis_offset=0.
            
        es.paper_format=paper_format
        
        if not lcolor is None:
            s.lcolor=lcolor
        es.color_pigmodes=color_pigmodes
        es.color_theo_modes=color_theo_modes
        es.Emodes_legend=legend
        
        es.vMinMax=vMinMax
        es.add_streched=add_streched
        es.amp_as_markersize=amp_as_markersize
        es.minmarkersize=minmarkersize
        es.markersize_pigmodes=markersize_pigmodes
        
        es.plot_streched_theo_modes=plot_streched_theo_modes
        es.frequency_dependent_eps_q=frequency_dependent_eps_q
        es.keep_ploting_theo_modes=False
        es.pause_redraw=False
        
        if isinstance(s.fminmax, bool):
            s.set_fminmax(s.fminmax,double=es.double)
        
        label= 'PSD [ppm²/μHz]'
        if sqrt_psd:
            label= fr'$\sqrt{{{label}}}$'
            s.psd_sm=np.sqrt(s.psd_sm)
            
        if isinstance(ax_echelle,bool):
            es.fig=plt.figure(figsize= (10,6))
            es.ax_echelle=es.fig.gca()
            es.fig.subplots_adjust(top=0.90,bottom=0.1,left=0.07,right=0.94,hspace=0.3,wspace=4)
        else:
            es.ax_echelle=ax_echelle
            es.fig = ax_echelle.get_figure()
        
        if colorbar:
            Gridspec=es.ax_echelle.get_subplotspec().subgridspec(1,30,wspace=0)
            es.ax_echelle.remove()
            es.ax_echelle = plt.subplot(Gridspec[0, :28])
            colorbar = plt.subplot(Gridspec[0, 28:29])
        
        if add_streched:
            strechnames=['dpi','q','eps_p1_init','eps_g1']
            strechvars= [dpi,q,eps_p1,eps_g1]
            
            for i in range(len(strechnames)):
                if not hasattr(s,strechnames[i]):
                    setattr(s,strechnames[i],float(strechvars[i]))
            
            es.fig.set_size_inches(14, 6) 
            Gridspec=es.ax_echelle.get_subplotspec().subgridspec(1,3,wspace=0)
            es.ax_echelle.remove()
            es.ax_echelle        = plt.subplot(Gridspec[0,0:1])
            es.ax_strechedPeriod = plt.subplot(Gridspec[0, 1:2])
            es.ax_strechedFreq   = plt.subplot(Gridspec[0, 2:3])
            
            es.ax_strechedPeriod.tick_params(labelleft=False,left=False,labelright=False,right=False)
            es.ax_strechedFreq.tick_params(  labelleft=False,left=False,labelright=True,right=True)
            es.ax_strechedFreq.yaxis.set_label_position("right")
            
            es.ax_strechedPeriod.sharey(es.ax_echelle)
            es.ax_strechedFreq.sharey(es.ax_echelle)
            
        if hasattr(s, 'modes2'):
            s.es.set_modes_to_0()
            
        if not es.frequency_dependent_eps_q and 'pi_modes_init' in s.__dict__:
            es.pi_modes_init_temp=s.pi_modes_init
            es.eps_p1_init_temp=s.eps_p1_init
            es.freq_minmax_to_Fit_temp=s.freq_minmax_to_Fit
            delattr(s,'pi_modes_init')
            s.eps_p1_init=np.mean(es.eps_p1_init_temp)
        
            es.q_temp=s.q
            s.q=np.mean(es.q_temp)
            
            delattr(s,'freq_minmax_to_Fit')
            delattr(s,'gamma_modes_theo')
            
        if s.acoustic_glitch and 'pi_modes_init' in s.__dict__:
            delattr(s, 'pi_modes_init')
         
        es.redraw()
        
        es.ax_echelle.set_ylabel('ν (μHz)')
        
        if colorbar:
            cbar= es.fig.colorbar(es.cm_echelle,cax=colorbar,shrink=2)#,pad=0.01)
            cbar.set_label(label)
            
    def __getattr__(es, name):
        if name == "psd_no02":
            return s.psd_sm
        raise AttributeError(f"{name} not found")
        
        
    def echelle_cm(es,echelle_type='frequency'):
        s=es.s
        
        n=es.resolution
        
        sign=1
        flip=np.s_[:]
        match echelle_type:
            case 'frequency':
                x = s.freq_sm
                y = s.freq_sm
                z = s.psd_sm
                deltaX=s.dnu
                ax=es.ax_echelle
                ax.set_xlabel('ν mod {:3.2f} (μHz)'.format(s.dnu))
                
            case 'streched period':
                if hasattr(es, 'q_func'):
                    q=es.q_func(s.freq_sm)
                    # eps_p1=es.eps_p1_init_func(s.freq_sm)
                else:
                    q=np.mean(s.q)
                    eps_p1=np.mean(s.eps_p1_init)
                    
                if es.frequency_dependent_eps_q and hasattr(s,'gamma_modes_theo'):# and not s.discrete_orders:
                    s.tau = calc_streched_period(freq=s.freq_sm,q=q,pi_modes=s.pi_modes_theo,dnu=s.dnu,dpi=s.dpi
                                                ,acoustic_glitch=s.acoustic_glitch_phase,A_p=s.A_p_glitch,tau_p=s.tau_p_glitch,phi_p=s.phi_p_glitch,b_p=s.b_p_glitch)
                else:
                    s.tau = calc_streched_period(freq=s.freq_sm,q=q,eps_p1=eps_p1,dnu=s.dnu,dpi=s.dpi)
                    
                y = s.freq_sm
                z = es.psd_no02
                deltaX=s.dpi
                ax=es.ax_strechedPeriod
                ax.set_xlabel(f"τ mod {s.dpi:3.2f} (s)")
                sign=-1
                flip=np.s_[::-1]
                x=s.tau

            case 'streched frequency':
                if hasattr(es, 'q_func'):
                    q=es.q_func(s.freq_sm)
                    # eps_g1=es.eps_g1_func(s.tau)
                else:
                    q=np.mean(s.q)
                    eps_g1=np.mean(s.eps_g1)
                if es.frequency_dependent_eps_q and hasattr(s,'gamma_modes_theo') :#and not s.discrete_orders:
                    x = calc_streched_freq(freq=s.freq_sm,q=q,gamma_modes=s.gamma_modes_theo,dnu=s.dnu,dpi=s.dpi,glitch_type=s.glitch_type
                                          ,buoyancy_glitch=s.buoyancy_glitch_phase,A_g=s.A_g_glitch,omega_g=s.omega_g_glitch,phi_g=s.phi_g_glitch,Delta_g=s.Delta_g_glitch)
                else:
                    x = calc_streched_freq(freq=s.freq_sm,q=q,eps_g1=eps_g1,dnu=s.dnu,dpi=s.dpi)
                    
                y = s.freq_sm
                z = es.psd_no02
                deltaX=s.dnu
                ax=es.ax_strechedFreq
                ax.set_xlabel(f"f mod {s.dnu:3.2f} (μHz)")
                
        if es.double:
            ax.axvline(0,color='k',zorder=100)
            double=2
        else:
            double=1
        
        x_mod  = x%(deltaX)   
        idxs_newline,=np.where(sign*np.diff(x_mod%deltaX)<-0)
            
        num_lines=len(idxs_newline)-double+1
        
        
        # plt.figure()
        # plt.plot(x)
        # for i in idxs_newline:
        #     plt.axhline(x[i])
        
        # plt.figure()
        # plt.plot(x % deltaX)
        
        x_interp= np.linspace(0,deltaX*double,n)
        n_interp_stepsize=np.median(np.diff(x_interp))
        X       = np.linspace(0-n_interp_stepsize/2,deltaX*double+n_interp_stepsize/2,n+1)
        Y       = np.zeros(num_lines+1)
        Z       = np.zeros([num_lines,len(x_interp)])
        
        ##################
        # plt.figure()
        # ax_tot=plt.gca()
        # ax_tot.plot(x,z,'k')
        # for i in idxs_newline:
        #     ax_tot.axvline(x[i])
        
        # plt.figure()
        # ax_stacked=plt.gca()
        
        # plt.figure()
        # ax_test=plt.gca()
        ###################
        
        idx_start=0
        Y[0] = y[idx_start]
        for idx_line,idx_stop in enumerate(idxs_newline+1):
            if es.double and idx_line==num_lines:
                continue
            
            x_line=x[idx_start:idxs_newline[idx_line]+1]
        
            if es.double:
                # x_line1=x_line
                x_line2=x[idxs_newline[idx_line]+1:idxs_newline[idx_line+1]+1]
                tostack=[x_line[flip]%(deltaX),x_line2[flip]%(deltaX)][flip]
                tostack[1]=tostack[1]+deltaX
                x_line=np.hstack(tostack)
                # x_line_plot=np.hstack((x_line1[flip],x_line2[flip])[flip])
            else:
                # x_line_plot=x_line
                x_line=x_line[flip]%(deltaX)
                
            z_line=z[idx_start:idxs_newline[idx_line+double-1]+1][flip]
        
            Z[idx_line,:]= np.interp(x_interp,x_line,z_line)
            
            Y[idx_line+1] = y[idxs_newline[idx_line]+1]
            
            #########################
            # ax_tot.plot(x_line_plot,z_line,'--',lw=3)
              
            # ax_stacked.scatter((x_line),np.ones_like(x_line)*y[idxs_newline[idx_line+double-1]+1],c=z_line[flip]) 
            # ax_stacked.plot((x_line),np.linspace(Y[idx_line],Y[idx_line+1],len(x_line)),'o-',ms=1) 
            ######################
            
            idx_start=idx_stop
        
        es.cm_echelle = ax.pcolormesh(X,Y,Z,shading='flat',cmap=es.cmap,zorder=-100,vmin=es.vMinMax[0],vmax=es.vMinMax[1])
        # if es.double:
        #     ax.pcolormesh(X-deltaX,Y+s.dnu,Z,shading='flat',cmap=es.cmap,zorder=-100,vmin=es.vMinMax[0],vmax=es.vMinMax[1])
        
        # ax.set_xlim([(-s.xaxis_offset-deltaX*int(not bool(s.xaxis_offset)))*es.double ,  deltaX-s.xaxis_offset])
        
        offset=0
        if es.double and not s.xaxis_offset==0:
            if echelle_type=='streched period':
                if s.buoyancy_glitch and s.glitch_type=='gaussian' and hasattr(s, 'mixed_modes_theo'):
                    center,_=circular_mean(1e6/s.mixed_modes_theo%deltaX/deltaX)
                else:
                    center=np.mean(s.eps_g1)
                offset=(center-0.5)%1*deltaX
            else:
                offset=s.xaxis_offset/s.dnu*deltaX
            double=1
               
        ax.set_xlim(offset,  deltaX*double+offset)
        ax.set_ylim(s.fminmax[0],s.fminmax[1])
    
    def Emodes(es,echelle_type='',zorder=100,mew=None
                      ,modes=False,amp_as_markersize=None
                      ,color=False,marker=False,plot=True
                      ,deltaX=False,ax=False
                      ,save_streched_modes=False,plot_errorbars=True
                      ,legend=False,label=False):
        s=es.s
        if isinstance(modes,bool):
            modes=s.modes
        if echelle_type=='':
            xmodes=modes
            deltaX=deltaX or s.dnu 
            ax=ax or es.ax_echelle
        if not color:
            color=s.lcolor
        if not marker:
            marker=s.lmarker 
        
        if amp_as_markersize is None:
            amp_as_markersize=es.amp_as_markersize
        
        match echelle_type:
            case 'frequency':
                ax=es.ax_echelle
                xmodes=modes
                deltaX=s.dnu
                
            case 'streched period':
                if hasattr(es, 'q_func'):
                    q=es.q_func(modes)
                    # eps_p1=es.eps_p1_init_func(modes)
                else:
                    q=np.mean(s.q)
                    eps_p1=np.mean(s.eps_p1_init)
                ax=es.ax_strechedPeriod
                deltaX=s.dpi
                
                if es.frequency_dependent_eps_q and hasattr(s,'gamma_modes_theo'):# and not s.discrete_orders:
                    xmodes= calc_streched_period(freq=modes,q=q,pi_modes=s.pi_modes_theo,dnu=s.dnu,dpi=s.dpi
                                                ,acoustic_glitch=s.acoustic_glitch_phase,A_p=s.A_p_glitch,tau_p=s.tau_p_glitch,phi_p=s.phi_p_glitch,b_p=s.b_p_glitch)
                else:
                    xmodes= calc_streched_period(freq=modes, eps_p1=eps_p1, q=q,dnu=s.dnu,dpi=s.dpi)
                    
                if save_streched_modes:
                    s.gamma_modes_all_streched_sec=xmodes
                else:
                    s.gamma_modes_all_streched_sec_theo_temp=xmodes
                
            case 'streched frequency':
                if hasattr(es, 'q_func'):
                    q=es.q_func(modes)
                    # if save_streched_modes:
                    #     eps_g1=es.eps_g1_func(s.gamma_modes_all_streched_sec)
                    # else:
                    #     eps_g1=es.eps_g1_func(s.gamma_modes_all_streched_sec_theo_temp)
                else:
                    q=np.mean(s.q)
                    eps_g1=np.mean(s.eps_g1)
                ax=es.ax_strechedFreq
                deltaX=s.dnu
                
                if es.frequency_dependent_eps_q and hasattr(s,'gamma_modes_theo'):# and not s.discrete_orders:
                    xmodes = calc_streched_freq(freq=modes,q=q,gamma_modes=s.gamma_modes_theo,dnu=s.dnu,dpi=s.dpi,glitch_type=s.glitch_type
                                               ,buoyancy_glitch=s.buoyancy_glitch_phase,A_g=s.A_g_glitch,omega_g=s.omega_g_glitch,phi_g=s.phi_g_glitch,Delta_g=s.Delta_g_glitch)
                else:
                    xmodes = calc_streched_freq(freq=modes, eps_g1=eps_g1, q=q, dnu=s.dnu, dpi=s.dpi)
                
                if save_streched_modes:
                    s.pi_modes_all=xmodes
        
        if not plot:
            return xmodes
        if not isinstance(modes[0] ,(list,np.ndarray)):
            modes=np.array([modes])
            xmodes=np.array([xmodes])
            color=[color]
            marker=[marker]
                
        if isinstance(amp_as_markersize,bool) and amp_as_markersize:
            
            amp_min=min(s.modes_lor['amp_obs'].flatten())
            amp_max=max(s.modes_lor['amp_obs'].flatten())

            markersize=(s.modes_lor['amp_obs']-amp_min)/(amp_max-amp_min)*80+es.minmarkersize
        else:
            if isinstance(amp_as_markersize,bool):
                amp_as_markersize=20
            markersize=np.zeros_like(modes)+amp_as_markersize
        
        for l in range(len(modes)):
            if len(modes[l])==0:
                continue
            
            if not label:
                labell=f'l = {s.lList[l]}'
            else:
                labell=label
            if es.paper_format and l==1:
                markersize+=40
                
            if es.double and echelle_type=='frequency':
                n=modes[l]//s.dnu
                y_modes=(n+0.5)*s.dnu
            else:
                y_modes=modes[l]
            
            ax.scatter(xmodes[l] % deltaX,y_modes,s=markersize[l],marker=marker[l],color='k',facecolor=color[l],zorder=zorder,linewidths=mew,label=labell)
            # ax.plot(xmodes[l] % deltaX,modes[l],color=color[l],zorder=zorder)
            if legend:
                ax.legend(loc='upper center',ncol=len(modes[l]),facecolor='lightgray', framealpha=0.5, edgecolor=None)#,bbox_to_anchor=(0.04, 1))
            
            if echelle_type == 'frequency' and plot_errorbars:
                ax.errorbar(xmodes[l] % deltaX,y_modes,xerr=s.modes_sig[l],ecolor=color[l],zorder=zorder-1,ls='None',ms=0,capsize=5,barsabove=True)
               
            if es.double:
                ax.scatter(xmodes[l] % deltaX+deltaX,y_modes-s.dnu*(echelle_type=='frequency'),s=markersize[l],marker=marker[l],color='k',facecolor=color[l],zorder=zorder,linewidths=mew)
                if echelle_type == 'frequency' and plot_errorbars:
                    ax.errorbar(xmodes[l] % deltaX+deltaX,y_modes-s.dnu*(echelle_type=='frequency'),xerr=s.modes_sig[l],ecolor=color[l],zorder=zorder-1,ls='None',ms=0,capsize=5,barsabove=True)
                 
    #%%% interactive
    def redraw(es):
        s=es.s
        
        if es.pause_redraw:
            return
        
        if isinstance(es.vMinMax, bool):
            if es.vMinMax:
                es.vMinMax=[0,max(s.psd_sm)*0.3]
                # es.vMinMax=[0,np.median(s.psd_sm)]
            else:
                es.vMinMax=[None,None]     
        
        for artist in list(es.ax_echelle.lines) + list(es.ax_echelle.collections):
            artist.remove()
            
        es.echelle_cm(echelle_type='frequency')
        
        if hasattr(s, 'modes'):
            es.Emodes(echelle_type='frequency',plot_errorbars=not es.paper_format,legend=es.Emodes_legend)
            
        if es.add_streched:
            for lines in (list(es.ax_strechedPeriod.collections) +list(es.ax_strechedPeriod.lines) 
                        + list(es.ax_strechedFreq.collections) + list(es.ax_strechedFreq.lines)):
                lines.remove()
            
            if es.frequency_dependent_eps_q:
                if hasattr(s, 'pi_modes_init') and not s.acoustic_glitch:
                    s.eps_p1_init = s.pi_modes_init%s.dnu/s.dnu
                    pi_modes_to_interpolate=s.pi_modes_init
                else:
                    pi_modes_to_interpolate=False
                    
                s.pi_modes_theo = calc_pi_modes_theo(dnu=s.dnu,eps_p1=s.eps_p1_init,fminmax=s.freq_minmax_to_Fit,pi_modes_init=pi_modes_to_interpolate
                                                     ,acoustic_glitch=s.acoustic_glitch_freq,A_p=s.A_p_glitch,tau_p=s.tau_p_glitch,phi_p=s.phi_p_glitch,b_p=s.b_p_glitch)
                
                if (isinstance(s.eps_g1,np.ndarray) and not len(s.pi_modes_init)==len(s.eps_g1)):
                    pi_modes_to_calc_gamma_modes=1/s.mean_gamma_modes_sec*1e6
                elif not isinstance(s.eps_g1,np.ndarray):
                    pi_modes_to_calc_gamma_modes=False
                else:
                    pi_modes_to_calc_gamma_modes=s.pi_modes_init
                    
                s.gamma_modes_theo=calc_gamma_modes_theo(dpi_Ms=s.dpi_Ms,eps_g1=s.eps_g1,freq_minmax=s.freq_minmax_to_Fit,pi_modes=pi_modes_to_calc_gamma_modes,interp_kind=s.interp_kind
                                                         ,buoyancy_glitch=s.buoyancy_glitch_freq,glitch_type=s.glitch_type,A_g=s.A_g_glitch,omega_g=s.omega_g_glitch,phi_g=s.phi_g_glitch,Delta_g=s.Delta_g_glitch)
                # s.gamma_modes_theo=s.gamma_modes_theo[:3]
                if isinstance(s.q,(np.ndarray,list)):
                    if not len(s.q)==len(s.pi_modes_init):
                        print('q has wrong length. set to mean(q)')
                        s.q= np.array([np.mean(s.q)]*len(s.pi_modes_init))
                    es.q_func          =make_func_accept_Variables(
                        interp1d(s.pi_modes_init,s.q,kind='nearest',bounds_error=False,fill_value=(s.q[0],s.q[-1])))
                # es.eps_p1_init_func=make_func_accept_Variables(
                    # interp1d(s.pi_modes_init,s.eps_p1_init,kind=s.interp_kind,bounds_error=False,fill_value=(s.eps_p1_init[0],s.eps_p1_init[-1])))
                
                if es.estimate_eps_using_modes1:
                    color_pi_modes_init='red'
                else:
                    color_pi_modes_init=es.color_pigmodes[0]
                    
                es.Emodes(modes=s.pi_modes_init,amp_as_markersize=es.markersize_pigmodes,color=color_pi_modes_init,marker='o')
                if not hasattr(s, 'modes1') and not es.estimate_eps_using_modes1:
                    es.ax_strechedFreq.scatter(s.pi_modes_init%s.dnu, s.pi_modes_init, color=es.color_pigmodes[0],ec='k',s=es.markersize_pigmodes,zorder=90,marker='o',ls='None',label='l=1 (π modes)')
                    if es.Emodes_legend:
                        es.ax_strechedFreq.legend(loc='upper center',facecolor='lightgray', framealpha=0.5, edgecolor=None)
                    
            es.echelle_cm(echelle_type='streched period')
            if hasattr(s, 'modes') and not es.paper_format:
                es.Emodes(echelle_type='streched period',save_streched_modes=True)
                
            if es.frequency_dependent_eps_q:
                if hasattr(s,'eps_p1_init') and ((hasattr(s, 'modes1') and es.estimate_eps_using_modes1) or (isinstance(s.eps_g1,(np.ndarray,list)) and len(s.eps_g1>1) ) ):
                        
                    if es.estimate_eps_using_modes1:
                        if not hasattr(s, 'modes1'):
                            print('need modes to calc eps_g1, using init value')
                        else:
                            s.mean_gamma_modes_sec,s.mean_gamma_modes_sec_sig,s.eps_g1=es.get_mean_modes(modes1=s.modes1,modes1_sig=s.modes1_sig
                                                                            ,streched_modes1=s.gamma_modes_all_streched_sec[1],deltaX=s.dpi)
                            
                            es.ax_strechedPeriod.errorbar(s.eps_g1*s.dpi, 1/s.mean_gamma_modes_sec*1e6 ,xerr=s.mean_gamma_modes_sec_sig, color=es.color_pigmodes[1],zorder=100,marker='o',capsize=4,ls='None')
                    
                    # if not hasattr(s,'mean_gamma_modes_sec'):
                    #     print('no mean gamma mode found, calculated with eps_p1_init')
                    #     ng  = (1/(s.pi_modes_init*1e-6)) // s.dpi
                    #     s.mean_gamma_modes_sec= (s.eps_g1+ng) * s.dpi
                    
                    if len(s.eps_g1)==len(s.pi_modes_init):
                        y=s.pi_modes_init
                    else:
                        y=1/s.mean_gamma_modes_sec*1e6

                    es.ax_strechedPeriod.scatter(s.eps_g1*s.dpi, y, color=es.color_pigmodes[1],ec='k',zorder=100,marker='o',ls='None',s=es.markersize_pigmodes)
                     
                #     es.eps_g1_func=make_func_accept_Variables(
                #         interp1d(1/y*1e6,s.eps_g1,kind=s.interp_kind,bounds_error=False,fill_value=(s.eps_g1[-1],s.eps_g1[0])))
                # else:
                #     es.eps_g1_func=lambda x: s.eps_g1*np.ones_like(x)
                #     if not s.buoyancy_glitch:
                #         print('no eps_g1 found, using init value')
                
            es.echelle_cm(echelle_type='streched frequency')
            if hasattr(s, 'modes'):
                if not es.paper_format:
                    es.Emodes(echelle_type='streched frequency',save_streched_modes=True)
                if  hasattr(s, 'modes1') and es.frequency_dependent_eps_q and hasattr(s,'eps_p1_init'):
                    if es.estimate_eps_using_modes1:
                        s.pi_modes,pi_modes_sig,s.eps_p1=es.get_mean_modes(modes1=s.modes1,modes1_sig=s.modes1_sig
                                                                          ,streched_modes1=s.pi_modes_all[1],deltaX=s.dnu)
                        s.pi_modes_sig=pi_modes_sig
                    elif hasattr(s,'pi_modes_init_uncertainties'):
                        pi_modes_sig=s.pi_modes_init_uncertainties
             
                    es.ax_strechedFreq.errorbar(  s.pi_modes   %s.dnu, s.pi_modes,xerr=pi_modes_sig, color=es.color_pigmodes[0],zorder=100,marker='o',mec='k',capsize=4,ls='None',ms=9)
                    es.ax_echelle.errorbar(       s.pi_modes   %s.dnu, s.pi_modes,xerr=pi_modes_sig, color=es.color_pigmodes[0],zorder=100,marker='o',mec='k',capsize=4,ls='None',ms=9)
            
            if s.dnu_rot:
                es.plot_dtau_rot()
             
        if es.add_streched:
            if (not es.frequency_dependent_eps_q or not isinstance(s.eps_p1_init,(np.ndarray))) and not es.paper_format:
                es.ax_echelle.axvline(np.mean(s.eps_p1_init)*s.dnu,color='lightblue')
                if es.double:
                    es.ax_echelle.axvline(np.mean(s.eps_p1_init)*s.dnu+s.dnu,color='lightblue')
            # elif not isinstance(s.eps_p1_init,(np.ndarray)) and not es.paper_format:
            #     es.ax_echelle.axvline(s.eps_p1_init*s.dnu,color='lightblue')
            #     if es.double:
            #         es.ax_echelle.axvline(s.eps_p1_init*s.dnu+s.dnu,color='lightblue')
                    
            if not es.frequency_dependent_eps_q or (not hasattr(s,'modes1') and not isinstance(s.eps_g1, np.ndarray)) and not es.paper_format:
                es.ax_strechedPeriod.axvline(np.mean(s.eps_g1)*s.dpi,color='pink')
                if es.double:
                    es.ax_strechedPeriod.axvline(np.mean(s.eps_g1)*s.dpi+s.dpi,color='pink')
    
        if (s.buoyancy_glitch or s.acoustic_glitch) and es.frequency_dependent_eps_q and hasattr(s,'mixed_modes_theo') or es.keep_ploting_theo_modes:
            es.plot_theo_modes()
        
        es.fig.canvas.draw_idle()
        # es.fig.canvas.flush_events()
       
    def interactive(es,save_button=True):
        s=es.s
        width = 0.06
        height = 0.03
        xaxis_offset=0.13
        yoffset=0.
        xpos = 0.07
        ypos = 0.95
        labels=['Δν','ΔΠ','q','ε$_p$','ε$_g$','δν$_{rot}$']
        
        es.round_val=5
        
        if save_button:
            es._add_save_button()
        
        es.fig.subplots_adjust(top=0.85)
        if es.frequency_dependent_eps_q:
            es.fig.subplots_adjust(left=0.1)
            
        if s.buoyancy_glitch or s.acoustic_glitch:
            ypos_glitch=0.7
            xpos_glitch=0.92
            yoffset_glitch=0.05
            
            es.fig.subplots_adjust(right=0.85)
            
            labels_glitch=[]
            es.glitch_variable_names=[]
            es.glitch_steps=[]
                
            if s.acoustic_glitch:
                labels_glitch+=['A$_p$','τ$_p$','φ$_p$','b$_p$']
                es.glitch_variable_names+=["A_p_glitch","tau_p_glitch","phi_p_glitch","b_p_glitch"]
                es.glitch_steps+=[0.03,0.003,0.1,1e-4]
                
            if s.buoyancy_glitch:
                labels_glitch+=['A$_g$','ω$_g$','φ$_g$']
                es.glitch_variable_names+=["A_g_glitch","omega_g_glitch","phi_g_glitch"]
                
                if s.glitch_type=='gaussian':
                    labels_glitch+=['Δ$g$']
                    es.glitch_variable_names+=['Delta_g_glitch']
                    A_step=10
                    delta_step=[1]
                else:
                    A_step=0.3
                    delta_step=[]
                es.glitch_steps+=[A_step,20,0.1,*delta_step]
                
        if es.add_streched:
            if es.frequency_dependent_eps_q and not s.buoyancy_glitch and not s.acoustic_glitch:
                if (es.estimate_eps_using_modes1 and not isinstance(s.eps_p1_init,(np.ndarray,list))):
                    es.variable_names=["dnu","dpi","q","eps_p1_init"]
                    labels=labels[:4]
                else:
                    es.variable_names=["dnu","dpi"]
                    labels=labels[:2]
                
            else:
                es.variable_names=["dnu","dpi","q","eps_p1_init","eps_g1"]
                labels=labels[:5]
                
            if (s.acoustic_glitch or s.buoyancy_glitch) and es.frequency_dependent_eps_q:
                es.variable_names.remove("q")     
                labels.remove('q')
                
            es.onIoff_epsp=False
            es.onIoff_epsg=False
        else:
            es.variable_names=["dnu"]
            labels=labels[:1]
            
        if s.dnu_rot:
            es.variable_names.append("dnu_rot")
            labels.append('δν$_{rot}$')
            
        if es.double:
            es.variable_names.append("xaxis_offset")
            labels.append("offset")
            
        s.variables=Variables(s=s,variable_names=es.variable_names)
        es.boxes=[0]*len(s.variables)
        
        if s.buoyancy_glitch or s.acoustic_glitch:
            s.glitch_variables=Variables(s=s,variable_names=es.glitch_variable_names)
            es.boxes_glitch=[0]*len(s.glitch_variables)
        
        es.steps = [0.1] * len(s.variables)
        es.boxes_step=[0]*len(s.variables)
        for idx_box in range(len(s.variables)):
            es.boxes[idx_box] =      TextBox(ax=es.fig.add_axes([xpos+xaxis_offset*idx_box, ypos-yoffset*idx_box, width, height])
                                      ,label=labels[idx_box]+' = ',initial=round(np.mean(s.variables[idx_box]),es.round_val))
            
            es.boxes_step[idx_box] = TextBox(ax=es.fig.add_axes([xpos+xaxis_offset*idx_box, ypos-0.05-yoffset*idx_box, width, height])
                                      ,label='step = ',initial=es.steps[idx_box])
            
            es.boxes[idx_box].on_submit(     lambda boxval,idx_box=idx_box: es._update_textbox(idx_box=idx_box, boxval=boxval))
            es.boxes_step[idx_box].on_submit(lambda boxval,idx_box=idx_box: es._update_step(   idx_box=idx_box, boxval=boxval))
            
        es.fig.canvas.mpl_connect('key_press_event', lambda event: es._onclick_keyboard_arrows(event))
        
        if s.buoyancy_glitch or s.acoustic_glitch:
            for idx_box in range(len(s.glitch_variables)):
                es.boxes_glitch[idx_box] =      TextBox(ax=es.fig.add_axes([xpos_glitch, ypos_glitch-yoffset_glitch*idx_box, width, height])
                                          ,label=labels_glitch[idx_box]+' = ',initial=round(np.mean(s.glitch_variables[idx_box]),es.round_val))
                es.boxes_glitch[idx_box].on_submit(     lambda boxval,idx_box=idx_box: es._update_glitch_textbox(idx_box=idx_box, boxval=boxval))
        
        if es.add_streched:
            es.add_add_epsp_button()
            es.add_theo_modes_button()
            es.peakbag_full_psd_button()
            es._template_button()
            es.fit_template_button()
            if es.frequency_dependent_eps_q and not hasattr(s,'modes1'):
                es._add_add_epsg_button()
        
        if es.add_streched and es.frequency_dependent_eps_q and hasattr(s,'pi_modes_init'):
            es._got_eps_q()
            
    def _update_textbox(es,idx_box=False,boxval=False):
        s=es.s
        
        es.active_box_group = "global_boxes"
        es.idx_active_box = idx_box
        
        s.variables[idx_box]=float(boxval) 
        es.idx_active_box=idx_box
        
        if idx_box==0 and hasattr(s,'modes1'):
            s.get_np1()

        es.redraw()
        
    def _update_glitch_textbox(es,idx_box=False,boxval=False):
        s=es.s
        
        es.active_box_group = "glitch_boxes"
        es.idx_active_box = idx_box
        
        s.glitch_variables[idx_box]=float(boxval) 
        es.idx_active_box=idx_box

        es.redraw()
        
    def _update_step(es, idx_box, boxval):
        es.active_box_group = "global_boxes"
        es.idx_active_box = idx_box
        
        es.steps[idx_box] = float(boxval)
        
    def _onclick_keyboard_arrows(es,event):
        if event.key=="up":
            sign   = 1
        elif event.key=="down":
            sign = -1
        else:
            return 
        s=es.s
        
        group = es.active_box_group
        idx = es.idx_active_box
        
        if group == "global_boxes":
            step = es.steps[idx]  
        elif group == "glitch_boxes":
            step=es.glitch_steps[idx]
        else:
            step=0.01
        upIdown = step * sign
        
        if group == "global_boxes":
            s.variables[idx] += upIdown
            es.boxes[idx].set_val(str(round(s.variables[idx], es.round_val)))
        
        elif group == "glitch_boxes":
            s.glitch_variables[idx] += upIdown
            es.boxes_glitch[idx].set_val(str(round(s.glitch_variables[idx], es.round_val)))

        elif group == "eps_boxes":
            s.eps_p1_init[idx] += upIdown
            s.pi_modes_init[idx] = (s.eps_p1_init[idx] * s.dnu + (s.pi_modes_init[idx] // s.dnu * s.dnu))
           
            es.eps_boxes[idx].set_val(str(round(s.eps_p1_init[idx], es.round_val)))
            
        elif group == "q_boxes":
            s.q[idx] += upIdown
            es.q_boxes[idx].set_val(str(round(s.q[idx], es.round_val)))
            
    def _update_box_after_fit(es):
        s=es.s
        es.pause_redraw=True
        for i_n in range(len(es.q_boxes)):
            es.q_boxes[i_n].set_val(str(round(s.q[i_n], es.round_val)))
            if not s.acoustic_glitch:
                es.eps_boxes[i_n].set_val(str(round(s.eps_p1_init[i_n], es.round_val)))
            
        if s.buoyancy_glitch or s.acoustic_glitch:
            for i_n in range(len(es.boxes_glitch)):
                es.boxes_glitch[i_n].set_val(str(round(s.glitch_variables[i_n],es.round_val)))
            
            if s. acoustic_glitch:
                idx_epsp_box,= np.where(np.array(es.variable_names)=='eps_p1_init')[0]
                es.boxes[idx_epsp_box].set_val(str(round(s.variables[idx_epsp_box],es.round_val)))
                
            if s.buoyancy_glitch:
                idx_epsg_box,= np.where(np.array(es.variable_names)=='eps_g1')[0]
                es.boxes[idx_epsg_box].set_val(str(round(s.variables[idx_epsg_box],es.round_val)))
            
        if s.fit_dpi:
            idx_dpi_box,= np.where(np.array(es.variable_names)=='dpi')[0]
            es.boxes[idx_dpi_box].set_val(str(round(s.variables[idx_dpi_box],es.round_val)))
        
        es.pause_redraw=False
        es.redraw()
          
    #%%% eps&co buttons
    def add_add_epsp_button(es):
        ax_button = es.fig.add_axes([0.72, 0.925, 0.06, 0.03])
        es._add_initial_epsp_button = Button(ax_button, r'add $ε_{p,init}$')
        es._add_initial_epsp_button.on_clicked(lambda event: es._add_initial_epsp(event))
        
    def _add_add_epsg_button(es):
        ax_button = es.fig.add_axes([0.72, 0.895, 0.06, 0.03])  #[left, bottom, width, height]
        es._add_initial_epsg_button = Button(ax_button, r'add $ε_{g,init}$')
        es._add_initial_epsg_button.on_clicked(lambda event: es._add_initial_epsg(event))
              
    def _add_initial_epsp(es, event):
        s=es.s
        es.onIoff_epsp = not es.onIoff_epsp
        
        if hasattr(es,'pi_modes_init_temp'):
            s.pi_modes_init=es.pi_modes_init_temp
            s.eps_p1_init=es.eps_p1_init_temp
            s.q=es.q_temp
            s.freq_minmax_to_Fit=es.freq_minmax_to_Fit_temp
            delattr(es, 'pi_modes_init_temp')
            delattr(es, 'eps_p1_init_temp')
            delattr(es, 'q_temp')
            delattr(es, 'freq_minmax_to_Fit_temp')
            
        es.frequency_dependent_eps_q=True
        
        if es.onIoff_epsp:
            if len(es.boxes)>2 and not hasattr(es,'eps_boxes'):
                if not isinstance(s.eps_g1,np.ndarray):
                    idx_boxes_to_remove= [i for i, element in enumerate(es.variable_names) if element in ['q','eps_p1_init']]
                else:
                    idx_boxes_to_remove= [i for i, element in enumerate(es.variable_names) if element in ['q','eps_p1_init','eps_g1']]
                for i in idx_boxes_to_remove:
                    es.fig.subplots_adjust(left=0.1)
                    es.boxes[i].disconnect_events()
                    es.boxes[i].ax.remove() 
                    es.boxes_step[i].disconnect_events()
                    es.boxes_step[i].ax.remove()
                    # s.variables=Variables(s=s,variable_names=es.variable_names[:2])
            
            es.restart_eps= hasattr(es,'eps_boxes')
            
            if es.restart_eps:
                for i in range(len(es.eps_boxes)): es.eps_boxes[i].disconnect_events() 
                for i in range(len(es.eps_boxes)): es.eps_boxes[i].ax.remove() 
                for i in range(len(es.eps_boxes)):   es.q_boxes[i].disconnect_events() 
                for i in range(len(es.eps_boxes)):   es.q_boxes[i].ax.remove() 
                
            es.eps_boxes = []
            es.q_boxes = [] 
            
            if "pi_modes_init" in s.__dict__ and es.add_streched and not es.restart_eps:
                es._got_eps_q()
                es.onIoff_epsp = False
            else:
                es._add_initial_epsp_button.label.set_color("green")
                
                s.pi_modes_init=np.array([])
                s.eps_p1_init=np.array([])
                es.interactive_epsp_on=es.fig.canvas.mpl_connect('button_press_event', lambda event: es._onclick_get_eps_p(event))
            
                es._add_fit_button()
            
            if not hasattr(es.s,'modes1') or not es.estimate_eps_using_modes1:
                es._add_add_epsg_button()
        
        else:
            es._add_initial_epsp_button.label.set_color("black")
            if hasattr(es,'interactive_epsp_on'):
                es.fig.canvas.mpl_disconnect(es.interactive_epsp_on)
                del es.interactive_epsp_on
            
        es.fig.canvas.draw_idle()
    
    def _add_initial_epsg(es, event):
        s=es.s
        es.onIoff_epsg = not es.onIoff_epsg
        
        if es.onIoff_epsg:
            es.onIoff_epsp=False
            es._add_initial_epsp_button.label.set_color("black")
            if hasattr(es,'interactive_epsp_on'):
                es.fig.canvas.mpl_disconnect(es.interactive_epsp_on)
                del es.interactive_epsp_on
            
            if len(es.boxes)>2 and not (isinstance(s.eps_g1,(list,np.ndarray)) and len(s.eps_g1)>1):
                if not hasattr(s, 'modes1') or not es.estimate_eps_using_modes1: 
                    idx_boxes_to_remove=[-1,0]
                for i in range(*idx_boxes_to_remove):
                    es.boxes[i].disconnect_events()
                    es.boxes[i].ax.remove() 
                    es.boxes_step[i].disconnect_events()
                    es.boxes_step[i].ax.remove()
                    
            es._add_initial_epsg_button.label.set_color("green")
            
            # es.epsg_boxes = []
            s.eps_g1=np.array([])
            s.mean_gamma_modes_sec=np.array([])
            es.interactive_epsg_on=es.fig.canvas.mpl_connect('button_press_event', lambda event: es._onclick_get_eps_g(event))
        
        else:
            es._add_initial_epsg_button.label.set_color("black")
            if hasattr(es,'interactive_epsg_on'):
                es.fig.canvas.mpl_disconnect(es.interactive_epsg_on)
                del es.interactive_epsg_on
                es.onIoff_epsg=False
        
        es.fig.canvas.draw_idle()
        
    def _onclick_get_eps_p(es,event):
        s=es.s
        if not event.inaxes == es.ax_echelle:
            return
        freq_y=event.ydata
        freq_x=event.xdata
        
        freq=freq_y//s.dnu * s.dnu + freq_x
        
        s.pi_modes_init=np.append(s.pi_modes_init,freq)
        s.eps_p1_init=np.append(s.eps_p1_init,freq_x/s.dnu)
        
        s.freq_minmax_to_Fit=[min(s.pi_modes_init)-s.dnu*0.45,max(s.pi_modes_init)+s.dnu*0.45]
        es._got_eps_q_i(freq)
        
    def _onclick_get_eps_g(es,event):
        s=es.s
        if not event.inaxes == es.ax_strechedPeriod:
            return
        
        streched_period_x = event.xdata
        freq_y            = event.ydata
        
        epsg=streched_period_x/s.dpi
 
        
        closest_pi_mode=s.pi_modes_init[np.argmin(abs(s.pi_modes_init-freq_y))]
        ng  = (1/(closest_pi_mode*1e-6)) // s.dpi
        s.mean_gamma_modes_sec=np.append(s.mean_gamma_modes_sec, (epsg+ng) * s.dpi)
        
        s.eps_g1=np.append(s.eps_g1,streched_period_x/s.dpi)
        
        sort_idx=np.argsort(s.mean_gamma_modes_sec)[::-1]
        s.mean_gamma_modes_sec=s.mean_gamma_modes_sec[sort_idx]
        s.eps_g1=s.eps_g1[sort_idx]
        
        es.redraw()
    
    def _got_eps_q(es):
        s=es.s
        
        es.eps_boxes = []
        es.q_boxes = [] 
        
        for freq in s.pi_modes_init:
            es._got_eps_q_i(freq)
          
        es._add_fit_button()
        
    def _got_eps_q_i(es,freq):
        s=es.s
        if not isinstance(s.q,(list,np.ndarray)):
            s.q=np.array([s.q])
        elif len(s.q)==len(es.eps_boxes):
            s.q=np.append(s.q,s.q[0])
        
        y0, y1 = es.ax_echelle.get_ylim()
        ypos = (freq - y0) / (y1 - y0)
        
        xpos  = 0.02
        width = 0.03
        height= 0.03
        
        if not s.acoustic_glitch:
            ax_box_eps = es.fig.add_axes([xpos, ypos+0.02, width, height])
            box_eps = TextBox(ax=ax_box_eps,label='$ε=$ ',initial=round((freq%s.dnu) / s.dnu, es.round_val))
            
        ax_box_q   = es.fig.add_axes([xpos, ypos-0.02, width, height])
        box_q = TextBox(ax=ax_box_q,label='$q=$ ',initial=round(s.q[len(es.q_boxes)], es.round_val))
        
        mode_id = freq 
        
        if not s.acoustic_glitch:
            def _update_epsp(new_val):
                idx = np.argmin(np.abs(np.array(s.pi_modes_init) - mode_id))
                es.active_box_group = "eps_boxes"
                es.idx_active_box = idx
                
                s.eps_p1_init[idx] = abs(float(new_val))
                s.pi_modes_init[idx] = (s.eps_p1_init[idx] * s.dnu + (s.pi_modes_init[idx] // s.dnu * s.dnu))
                
                es.redraw()
            
        def _update_q(new_val):
            idx = np.argmin(np.abs(np.array(s.pi_modes_init) - mode_id))
            es.active_box_group = "q_boxes"
            es.idx_active_box = idx
            
            s.q[idx] = abs(float(new_val))
            es.redraw()
        
        if not s.acoustic_glitch:
            box_eps.on_submit(_update_epsp)
            es.eps_boxes.append(box_eps)
            
        box_q.on_submit(_update_q)
        es.q_boxes.append(box_q)
        
        s.n_p = s.pi_modes_init//s.dnu
        s.num_orders=len(s.n_p)
        
        if len(s.pi_modes_init)==len(es.eps_boxes):
            sort_idx = np.argsort(s.pi_modes_init)
            s.pi_modes_init = s.pi_modes_init[sort_idx]
            s.eps_p1_init = s.eps_p1_init[sort_idx]
            s.q = s.q[sort_idx]
            s.n_p=s.n_p[sort_idx]
            
            if not s.acoustic_glitch:
                es.eps_boxes = [es.eps_boxes[i] for i in sort_idx]
            es.q_boxes   = [es.q_boxes[i] for i in sort_idx]
        
            es.redraw()
        
    def _add_save_button(es):
        ax_save = es.fig.add_axes([0.97, 0.02, 0.02, 0.02])  # [left, bottom, width, height]
        es.save_button = Button(ax_save, 'S', color='palegreen', hovercolor='0.85')
        def on_click_save(event):
            s = es.s
            s.save()
        es.save_button.on_clicked(on_click_save)
            
        
    #%%% mean_modes
    def get_mean_modes(es,modes1,modes1_sig,streched_modes1,deltaX,only_one_np=False):
        s=es.s

        if not only_one_np:
            unique_np = np.unique(s.np1)
        else:
            unique_np = [0]
            mask=modes1==modes1
            
        streched_modes1_mean     = np.zeros_like(unique_np, dtype=float)
        streched_modes1_mean_sig = np.zeros_like(unique_np, dtype=float)
        eps_x1                   = np.zeros_like(unique_np, dtype=float)
        
        for i,np_i in enumerate(unique_np):
            if not only_one_np:
                mask = s.np1 == np_i
            
            epsX_same_n = streched_modes1[mask] % deltaX /deltaX
            weights = 1 / np.array(modes1_sig)[mask]
            nX = np.mean(streched_modes1[mask]) // deltaX
            
            eps_x1[i],stddev_eps=circular_mean(epsX_same_n, weights=weights)
            streched_modes1_mean[i]     = (eps_x1[i]+nX) *deltaX
            if stddev_eps<1e-25:
                stddev_eps=1e-25 #!!!
            streched_modes1_mean_sig[i] = stddev_eps     *deltaX
            
        return streched_modes1_mean,streched_modes1_mean_sig,eps_x1
    
    #%%% simple fit strech
    def _add_fit_button(es):
        s=es.s
        ax_button = es.fig.add_axes([0.85, 0.895, 0.06, 0.03])  #[left, bottom, width, height]
        es.fit_button_quick = Button(ax_button, 'Fit q', color='orange', hovercolor='0.85')
    
        # ax_button = es.fig.add_axes([0.88, 0.895, 0.03, 0.03])  # [left, bottom, width, height]
        # es.fit_button_MCMC = Button(ax_button, 'MCMC', color='red', hovercolor='0.85')
        
        es.fit_button_quick.on_clicked(es._fit_button_clicked)
        # es.fit_button_MCMC.on_clicked(es._fit_button_MCMC_clicked)
        
    def _fit_button_clicked(es, event):
        s=es.s
        es.estimate_eps_using_modes1=True
        
        if not hasattr(s,'MCMC_or_dynesty'):
            s.MCMC_or_dynesty=False
        
        if s.MCMC_or_dynesty:
            s=es.s
            mixed_modes_Fit(s,MCMC_or_dynesty=s.MCMC_or_dynesty,fit_dpi=False)
            # new_button_color='green'
        
            # es.fit_button_MCMC.color = new_button_color
            # es.fit_button_MCMC.ax.set_facecolor(new_button_color)
            es._update_box_after_fit()
            
        else:
            
            es.find_best_epsp_q()
            # new_button_color='crimson'
            es._update_box_after_fit()
        
    def find_best_epsp_q(es):
        s=es.s
        
        results_simpleFit =np.zeros([len(np.unique(s.np1)),2])
        
        for i_np,npi in enumerate(np.unique(s.np1)):
            mask_n=s.np1==npi
            modes1_n     = s.modes1[mask_n]
            modes1_sig_n = s.modes1_sig[mask_n]
            
            initial_guess_eps_q=[s.pi_modes_init[i_np],s.q[i_np]]
            fct_to_minimize = es.make_calc_X_for_epsp_q_func(modes1_n,modes1_sig_n)
            
            
            bounds=[(0, None), (0., 0.7)]
            
            minimizer_options = {"method": "L-BFGS-B"
                                ,"bounds": bounds
                                ,"options": {"ftol": 1e-14, "gtol": 1e-10, "maxiter": 5000}
                                }
            
            if s.minimizer=='minimize':
                result_n=minimize(fct_to_minimize,initial_guess_eps_q,method=minimizer_options['method'],bounds=bounds,options=minimizer_options['options'])
                
            elif s.minimizer=='differential_evolution':
                result_n=differential_evolution(fct_to_minimize,bounds=bounds,polish=False,maxiter=100000)
                result_n=minimize(fct_to_minimize,result_n.x,method=minimizer_options['method'],bounds=bounds
                                  # ,options=minimizer_options['options']
                                  )

            elif s.minimizer == 'basinhopping':
                result_n = basinhopping(fct_to_minimize, initial_guess_eps_q,  minimizer_kwargs=minimizer_options
                                        # ,niter=1000
                                        )
            
            results_simpleFit[i_np,:]=result_n.x
            
        s.pi_modes_init=results_simpleFit[:,0]
        s.q            =results_simpleFit[:,1]
        
        s.eps_p1_init  =s.pi_modes_init % s.dnu / s.dnu
    
    def make_calc_X_for_epsp_q_func(es,modes1_n,modes1_sig_n):
        s = es.s
        def _calc_X_for_epsp_q_(pi_and_q):
            pi_mode_init_n,q_n=pi_and_q
            
            eps_p1_init_n=(pi_mode_init_n%s.dnu)/s.dnu
            # modes=s.modes[]
            
            gamma_modes_all_n=calc_streched_period(freq=modes1_n, eps_p1=eps_p1_init_n, q=q_n,dnu=s.dnu,dpi=s.dpi)
            gamma_mode_n,gamma_mode_n_sig,eps_g1_n=es.get_mean_modes(modes1=modes1_n,modes1_sig=modes1_sig_n
                                                            ,streched_modes1=gamma_modes_all_n,deltaX=s.dpi,only_one_np=True)
            
            pi_modes_all_n = calc_streched_freq(freq=modes1_n, eps_g1=eps_g1_n, q=q_n,dnu=s.dnu,dpi=s.dpi)
            pi_mode_n,pi_mode_sig_n,eps_p1_n=es.get_mean_modes(modes1=modes1_n,modes1_sig=modes1_sig_n
                                                            ,streched_modes1=pi_modes_all_n,deltaX=s.dnu,only_one_np=True)
            
            return (pi_mode_sig_n/s.dnu)**2+(gamma_mode_n_sig/s.dpi)**2+((pi_mode_init_n-pi_mode_n)/pi_mode_sig_n)**2 
        return _calc_X_for_epsp_q_

    #%%%% rot_split
    def calc_rotational_splitting_tau(es,freq,dnu_rot,m,dpi=False,dnu=False,numax=False): #gehan2018  
        s=es.s
        dnu=(dnu or s.dnu)*1e-6
        numax=(numax or s.numax)*1e-6
        dpi= dpi or s.dpi
        freq = freq*1e-6
        dnu_rot= dnu_rot*1e-6
        
        N=dnu/(s.dpi*(numax)**2)
        dtau_rot = s.dpi*(1+(2*m*N/(N+1)*dnu_rot/freq))
        
        return dtau_rot

    def plot_dtau_rot(es,color='m'):
        s=es.s
        ax=es.ax_strechedPeriod
        ax.autoscale(False)
        
        fmin=s.fminmax[0]
        fmax=s.fminmax[1]
            
        # if not hasattr(s,'eps_g1'):
        # eps_g1=1/2 #!!!
        
        freq_g=np.array((1/(s.dpi*np.array(range(1,1000)))))*1e6
        freq_g=freq_g[(freq_g<fmax+10) & (freq_g>fmin-10) ]

        for m in [-1,0,1]:
            N=s.dnu/(s.dpi*1e-6*(s.numax*1e-6)**2)
            freq_gi = (freq_g*1e-6-m*s.dnu_rot*1e-6*N/(N+1))*1e6
            
            pi_g = 1/(freq_gi*1e-6)
            
            dtau_rot=es.calc_rotational_splitting_tau(freq=freq_gi,dnu_rot=s.dnu_rot,m=m)

            for n in [-1,0,1]:
                #ax.plot(((pi_g)%dpi)         +dpi*n+eps_g1,freq_gi ,'.',ms=2,color='r')#,ls='-')
                #ax.plot(((dtau_rot)%dpi)     +dpi*n+eps_g1,freq_g,'.',ms=2,color="g")#,ls='-')
                ax.plot(((dtau_rot+pi_g)%s.dpi)+s.dpi*n+s.eps_g1*s.dpi,freq_gi,'.',ms=4,color="m")
                
                
    #%%% theo modes buttons
    
    def peakbag_full_psd_button(es):
        ax_button = es.fig.add_axes([0.85, 0.925, 0.06, 0.03])  #[left, bottom, width, height]
        es._peakbag_button = Button(ax_button, 'peakbag')
        es._peakbag_button.on_clicked(es._peakbag_button_clicked)
    
    def _peakbag_button_clicked(es,event):
        s=es.s
        s.spsd.fit_all_modes_at_once()
        es.redraw()
        
    def fit_template_button(es):
        ax_button = es.fig.add_axes([0.915, 0.925, 0.06, 0.03])  #[left, bottom, width, height]
        es._fit_template_button = Button(ax_button, 'fit template ')
        es._fit_template_button.on_clicked(es._fit_template_button_clicked)
    
    def _fit_template_button_clicked(es,event):
        s=es.s
        
        if s.discrete_orders:
            s.fit_template_discrete(MCMC_or_dynesty=s.MCMC_or_dynesty)
        else:
            s.fit_template_all()
        
        es._update_box_after_fit()

    def add_theo_modes_button(es):
        ax_button = es.fig.add_axes([0.785, 0.925, 0.06, 0.03])  #[left, bottom, width, height]
        es._theo_modes_button = Button(ax_button, 'theo modes')
        es._theo_modes_button.on_clicked(es._theo_modes_button_clicked)
        
    def _theo_modes_button_clicked(es, event):
        es.plot_theo_modes()
        es.fig.canvas.draw_idle()
        
    def plot_theo_modes(es):
        s=es.s
        
        if not es.frequency_dependent_eps_q and s.buoyancy_glitch:
            es.frequency_dependent_eps_q=True
            es.redraw()
        
        if s.discrete_orders and isinstance(s.q,(list,np.ndarray)):
            s.calc_theo_mixed_modes_discrete()
        else:
            s.mixed_modes_theo,s.Zeta,_=s.calc_theo_mixed_modes()

        es.Emodes(echelle_type='frequency',modes=s.mixed_modes_theo,amp_as_markersize=False
                          ,color=es.color_theo_modes,marker='x',plot_errorbars=False,zorder=10)
        if es.plot_streched_theo_modes:
            es.Emodes(echelle_type='streched period',modes=s.mixed_modes_theo,amp_as_markersize=False
                              ,color=es.color_theo_modes,marker='x',zorder=10,label='l=1 (template)',legend=es.Emodes_legend)
            es.Emodes(echelle_type='streched frequency',modes=s.mixed_modes_theo,amp_as_markersize=False
                              ,color=es.color_theo_modes,marker='x',zorder=10)
            
            # es.ax_echelle.plot(es.eps_p1_init_func(s.mixed_modes_theo)*s.dnu,s.mixed_modes_theo,'b+')
            if es.frequency_dependent_eps_q and not es.paper_format:
                es.ax_strechedPeriod.plot(1/s.gamma_modes_theo*1e6 % s.dpi,s.gamma_modes_theo,'+',color='pink')
                
                if s.acoustic_glitch_phase:
                    pi_modes_glitch=calc_pi_modes_theo(dnu=s.dnu,eps_p1=s.eps_p1_theo,fminmax=s.freq_minmax_to_Fit,pi_modes_init=s.pi_modes_theo
                                                      ,acoustic_glitch=s.acoustic_glitch,A_p=s.A_p_glitch,tau_p=s.tau_p_glitch,phi_p=s.phi_p_glitch,b_p=s.b_p_glitch)

                    es.ax_echelle.plot(pi_modes_glitch % s.dnu,pi_modes_glitch,'+',color='b',ms=10)
        
            
        # es.ax_echelle.plot(s.mixed_modes_theo  % s.dnu,s.mixed_modes_theo,"x",color='r',ms=4,zorder=100)
        
    def _template_button(es):
        ax_button = es.fig.add_axes([0.785, 0.895, 0.06, 0.03])  #[left, bottom, width, height]
        es._template_button = Button(ax_button, r'template')
        es._template_button.on_clicked(es.s.plot_template)

    #%%% 
    def set_modes_to_0(es,fwidth=0.05):
        s=es.s
        
        modes=np.concatenate((s.modes0, s.modes2))
        mask=np.any((np.abs(s.freq_sm[:,None] - modes) < fwidth*s.dnu),1)
        
        es.psd_no02=s.psd_sm.copy()
        es.psd_no02[mask]=0.
        
    
    def add_flippedPSD(es):
        s=es.s
        
        # [left, bottom, width, height]
        pos = es.axEchelle.get_position()
        es.axEchelle.set_position([0.1, pos.bottom, 0.3, pos.height])
        es.axSPeriod.set_position([0.4, pos.bottom, 0.3, pos.height])
        es.axSFreq.set_position(  [0.7, pos.bottom, 0.3, pos.height])
        
        es.axPSD.add_axis([0, pos.bottom, 0.1, pos.height])
        
        es.axPSD.sharey(es.axEchelle)

        
        plot_PSD(s.freq,s.psd,axPSD=es.axPSD,nu_max=s.numax,modes=s.modes,color_modes=s.lcolor,
                              zoom=0.2,flip=True)
        
        

#%% Stacked_psd
class Stacked_psd:
    #%%%
    def __init__(spsd,s,xaxis_offset=0,interactive=True,fwhm_constrain_type=False): #linear' or 'constant'
        if hasattr(s, 'es'):
            spsd.es=s.es
            es=s.es
            xaxis_offset=s.xaxis_offset
            es.fig.canvas.mpl_connect('button_press_event', lambda event: spsd.onclick_Echelle(event))
            
            if not hasattr(s,'_org_lcolor'):
                s._org_lmarker=s.lmarker
            s.lmarker=[s._org_lmarker[i] for i in s.lList]
            
        spsd.s=s
        s.spsd=spsd
        
        spsd.idx_l = 0
        
        if not hasattr(s,'_org_lcolor'):
            s._org_lcolor=s.lcolor
        s.lcolor=[s._org_lcolor[i] for i in s.lList]
        
    
        s.fwhm_constrain_type=fwhm_constrain_type
        
        freq_mod  = s.freq%(s.dnu)
        
        idxs_newline,=np.where(np.diff(freq_mod)<0)
        num_lines=len(idxs_newline)
        
        spsd.axes_stacked_psd=generate_subplot_grid((num_lines,1),hspace=0.25,bottom=0.05)[::-1]
    
        idx_start=0
        
        spsd.tol=int(len(s.freq)/100)
        for idx_line,idx_stop in enumerate(idxs_newline+1):
            
            n = len(s.freq)

            idx_min = max(0, idx_start - spsd.tol)
            idx_max = min(n - 1, idx_stop + spsd.tol)
            
            freq_min = s.freq[idx_min] - xaxis_offset
            freq_max = s.freq[idx_max] + xaxis_offset
                
            mask=(s.freq>freq_min) & (s.freq<freq_max)
            spsd.axes_stacked_psd[idx_line].plot(s.freq[mask],s.psd[mask],color='k',lw=0.8)
            spsd.axes_stacked_psd[idx_line].tick_params(axis='x', labelsize=5,pad=0)
            
            spsd.axes_stacked_psd[idx_line].set_xlim(s.freq[idx_start]-xaxis_offset,s.freq[idx_stop] -xaxis_offset)
            idx_start=idx_stop
            
        if interactive:
            fig=spsd.axes_stacked_psd[0].get_figure()
            fig.canvas.mpl_connect('button_press_event', lambda event: s._onclick_LorentzFit(event))
            fig.canvas.mpl_connect('key_press_event', spsd.onclick_spds)
            
    def onclick_spds(spsd,event):
        s=spsd.s
        
        if event.key == 'enter':
            
            l=s.lList[spsd.idx_l]
            s.fitted_lor.sort('x_0')
            
            if s.fwhm_constrain_type and not l==1:
                spsd.fit_modes_with_fwhm_constraint()
            
            setattr(s, f"modes{l}", s.fitted_lor['x_0'])
            setattr(s, f"modes{l}_lor", s.fitted_lor)
            setattr(s, f"modes{l}_sig", s.fitted_lor['sig'])
    
            names     = sorted([f"modes{i}"     for i in s.lList[:spsd.idx_l+1]])
            names_lor = sorted([f"modes{i}_lor" for i in s.lList[:spsd.idx_l+1]])
            names_sig = sorted([f"modes{i}_sig" for i in s.lList[:spsd.idx_l+1]])
            
            s.modes = Variables(s=s, variable_names=names)
            s.modes_lor = Variables(s=s, variable_names=names_lor)
            s.modes_sig = Variables(s=s, variable_names=names_sig)
            
            s.lList [:spsd.idx_l+1]   =sorted(s.lList[:spsd.idx_l+1])
            s.lcolor=[s._org_lcolor[i] for i in s.lList]
            
            delattr(s,'fitted_lor')
            
            if hasattr(s, 'es'):
                s.lmarker=[s._org_lmarker[i] for i in s.lList]
                
                s.got_modes()
                s.es.redraw()
                
            
            spsd.idx_l += 1
            
        else:
            s._onclick_delete_LorentzFit(event)
            
    def onclick_Echelle(spsd,event):
        s=spsd.s
        es=spsd.es
       
        ax= event.inaxes
        if not ax == es.ax_echelle:
            return

        freq_y=event.ydata
        freq_x=event.xdata
        
        freq=freq_y//s.dnu * s.dnu + freq_x
        
        print('{:.5f}, '.format(freq),end='')
        
        axes=spsd.axes_stacked_psd
        
        for ax in axes:
            xlim=ax.get_xlim()
            
            if freq>xlim[0] and freq<xlim[1]:
                ax.axvline(freq,color="r")
                ax_prev=ax
                
        fig=spsd.axes_stacked_psd[0].get_figure()
        # fig.canvas.manager.window.raise_()
        fig.canvas.draw_idle()#fig.canvas.renderer )
        fig.canvas.flush_events()
        
        
        if ax_prev:
            line=ax_prev.get_lines()[-1]
            line.remove()
                
    #%%% fit same fwhm
    def fit_modes_with_fwhm_constraint(spsd):
        s=spsd.s
    
        models=[]
        xs=[]
        ys=[]
        
        y=s.psd/s.y_to_scale
        
        for i_modes, modes_lori in enumerate(s.fitted_lor):
            xmin=modes_lori['xmin']
            xmax=modes_lori['xmax']
            mask_ToFit =(s.freq > xmin) & (s.freq < xmax)
            
            xi=s.freq[mask_ToFit]
            yi=y[mask_ToFit]
            
            Bkg_model=custom_model(lambda x: s.Bkg_func(x)/s.y_to_scale)()
            bkg=Bkg_model(xi)
            
            yi=yi+bkg
            
            l_init = astropy_models.Lorentz1D( x_0=(xmax-xmin)/2+xmin
                                      ,amplitude=np.mean(yi)
                                      ,bounds={"amplitude":(1e-9,5*max(s.psd)/s.y_to_scale)
                                               ,"x_0": (xmin,xmax)
                                               })
     
            l_init=l_init+Bkg_model
            
            models.append(l_init)
            xs.append(xi)
            ys.append(yi)
            
        if s.fwhm_constrain_type=='linear':
            meanFWHM= np.mean(s.fitted_lor['fwhm'])
            x_range=max(s.freq)-min(s.freq)
            
            fitter=JointChi2TwoDofFitter(joint_parameter_idx=[2],joint_parameter_p0=[0,meanFWHM],joint_parameter_bounds=[(0,10*meanFWHM/x_range),(-10*meanFWHM,10*meanFWHM)]
                                        ,joint_parameter_function=slope,model_x0_index=1,joint_para_always_positive=True
                                        ,minimizer=s.minimizer)

        else:
            fitter=JointChi2TwoDofFitter(joint_parameter_idx=[2],joint_parameter_p0=[np.mean(s.fitted_lor['fwhm'])],joint_parameter_bounds=[(0,5*max(s.psd)/s.y_to_scale)]
                                        ,minimizer=s.minimizer)

        lorentzians = fitter(models, xs,ys)
        
        if s.fwhm_constrain_type=='linear':
            if not hasattr(s,'fwhm_l0_func'):
                s._joint_parameter_result_l0_FWHM=fitter.joint_parameter_result
                s.fwhm_l0_func= lambda x: slope(x,*fitter.joint_parameter_result)
        
        for i_lor,lorentzi_with_Bkg in enumerate(lorentzians):
            lorentzi=lorentzi_with_Bkg[0]
            lorentzi.amplitude.value = lorentzi.amplitude.value * s.y_to_scale
            
            s.fitted_lor['x_0'][i_lor]     = lorentzi.x_0.value
            s.fitted_lor['amp_fit'][i_lor] = lorentzi.amplitude.value
            s.fitted_lor['fwhm'][i_lor]    = lorentzi.fwhm.value
            
            
            x_fit=np.linspace(s.fitted_lor['xmin'][i_lor],s.fitted_lor['xmax'][i_lor],1000)
            y_fit=lorentzi(x_fit)
            
            s.fitted_lor['sig'][i_lor]=s.get_sig_peakbag(lorentzi)
            
            x0=lorentzi.x_0.value
          
            for ax in spsd.axes_stacked_psd:
                x_data=ax.lines[0].get_xdata()
                if x0 > min(x_data) and x0 < max(x_data):
                    ax.plot(x_fit,y_fit,color='r',ls=':',dashes=[1,0.5]) 
                    # ax.axvline(x0,color=color,ls=':')
                    ax.axvline(x0,color='r',dashes=[1,0.5])

        plt.draw()
        
    def fit_all_modes_at_once(spsd,fix_fwhm=False,SNR_lim=2.5):
        s=spsd.s
        
        if not hasattr(s, 'modes0') and  hasattr(s, 'modes2'):
            print('peakbag l=0 and 2 modes first')
            
        if s.use_pre_fitted_l1_modes and not hasattr(s, 'modes1'):
            print('no modes1 found, used thoretical modes instead')
            s.use_pre_fitted_l1_modes=True
            
        marker_modeID     = ['|','|','|','|']
        markersize_modeID = [30,20,10,5]
        marker_ew_modeID  = [3,1.5,3,1]
            
        x=s.freq.copy()
        x_fit=np.linspace(x[0],x[-1],int(1e5))
            
        y=s.psd/s.y_to_scale
        
        Bkg_model=custom_model(lambda x: s.Bkg_func(x)/s.y_to_scale)()
        bkg=Bkg_model(x)
        y=y+bkg
        
        if not hasattr(s, "ax_peakbag"):
            plt.figure()
            s.ax_peakbag=plt.gca()
            s.ax_peakbag.plot(s.freq,s.psd+s.Bkg_func(s.freq),color='k',alpha=1,zorder=-10)
        
        s.generate_template(ax=s.ax_peakbag,inflateFWHM=s.inflateFWHM,color='cyan',use_pre_fitted_l1_modes=s.use_pre_fitted_l1_modes)

        # plt.plot(s.freq_sm,s.psd_sm+Bkg_model(s.freq_sm),color='k')
        # s.ax_peakbag.plot(x_fit,s.template_scaled(x_fit)*s.y_to_scale,color='cyan')
        
        fitter=Chi2TwoDofFitter(minimizer=s.minimizer)
        s.lorentzians_with_Bkg = fitter(s.template_scaled_init, x ,y)
        
        s.ax_peakbag.plot(x_fit,s.lorentzians_with_Bkg(x_fit)*s.y_to_scale,color='red')
        # s.ax_peakbag.plot(x_fit,s.Bkg_func(x_fit),color='cyan')
        
        len0=len(s.modes0_lor)
        len2=len0+len(s.modes2_lor)
        
        if not s.use_pre_fitted_l1_modes:
            s.modes1_lor=Table(names=('amp_obs','amp_fit','x_0','fwhm','xmin','xmax','sig'))
        for i_model in range(1,s.lorentzians_with_Bkg.n_submodels):  
            theo=''
            if i_model <= len0:
                l=0
                i_mode=i_model-1
            elif i_model <= len2:
                l=2
                i_mode=i_model-len0-1
            else:
                l=1
                i_mode=i_model-len2-1
                if not s.use_pre_fitted_l1_modes:
                    theo='_theo'
                
            lorentzi=s.lorentzians_with_Bkg[i_model]
            
            lorentzi.amplitude.value=lorentzi.amplitude.value*s.y_to_scale
            sig=s.get_sig_peakbag(lorentzi)
            
            getattr(s,f'modes{l}_lor'+theo)['x_0'    ][i_mode]=lorentzi.x_0.value
            getattr(s,f'modes{l}_lor'+theo)['amp_fit'][i_mode]=lorentzi.amplitude.value
            getattr(s,f'modes{l}_lor'+theo)['fwhm'   ][i_mode]=lorentzi.fwhm.value
            getattr(s,f'modes{l}_lor'+theo)['sig'    ][i_mode]=sig
            
            if l==1 and not s.use_pre_fitted_l1_modes:
                x_0=lorentzi.x_0.value
                xminb=x_0-s.freq_spacing_mixed_modes[i_mode]/2
                xmaxb=x_0+s.freq_spacing_mixed_modes[i_mode]/2
                mask_broad=(x > xminb) & (x < xmaxb)
            
                xmint=x_0-lorentzi.fwhm.value*2
                xmaxt=x_0+lorentzi.fwhm.value*2
                mask_tight=(x > xmint) & (x < xmaxt)
                
                height=min((s.modes1_lor_theo['amp_fit'][i_mode]+s.Bkg_func(s.modes1_lor_theo['x_0'][i_mode]))*0.9,np.max(y[mask_tight]*s.y_to_scale))
                idx_peaks,_=find_peaks(y[mask_broad]*s.y_to_scale, height=height, distance=len(x[mask_tight]))
                
                for i in range(len(idx_peaks)):
                    s.ax_peakbag.axvline(x[mask_broad][idx_peaks[i]],lw=0.7,color='black')
            
            xmin=getattr(s,f'modes{l}_lor'+theo)['xmin'][i_mode]
            xmax=getattr(s,f'modes{l}_lor'+theo)['xmax'][i_mode]
            
            if i_model==1:
                s.lorentzians = s.lorentzians_with_Bkg[1]
            else:
                s.lorentzians += s.lorentzians_with_Bkg[i_model]
            
            if hasattr(s, 'modes1') or not l==1 or ( (((s.modes1_lor_theo['amp_fit'][i_mode]+s.Bkg_func(s.modes1_lor_theo['x_0'][i_mode]))/s.Bkg_func(s.modes1_lor_theo['x_0'][i_mode])) > SNR_lim) and len(idx_peaks)<3) :
                if not hasattr(s, 'modes1') and l==1 and not np.abs(s.modes1_lor_theo['x_0'][i_mode]-(s.modes0[0]-s.dnu))<s.fwhm_l0_func(s.modes1_lor_theo['x_0'][i_mode])*5:
                    s.modes1_lor=vstack([s.modes1_lor, s.modes1_lor_theo[i_mode]])
                
                s.ax_peakbag.plot(lorentzi.x_0.value,np.max(s.psd),ls='None',marker=marker_modeID[l],ms=markersize_modeID[l],mew=marker_ew_modeID[l],color=s._org_lcolor[l])
                
                
            xi=np.linspace(xmin,xmax,1000)
            s.ax_peakbag.plot(xi,lorentzi(xi)+s.Bkg_func(xi),color=s._org_lcolor[l])
            
            for ax in spsd.axes_stacked_psd:
                x_data=ax.lines[0].get_xdata()
                if lorentzi.x_0.value > min(x_data) and lorentzi.x_0.value < max(x_data):
                    ax.plot(xi,lorentzi(xi),color='r',ls=':',dashes=[1,0.5]) 
                    # ax.axvline(x0,color=color,ls=':')
                    ax.axvline(lorentzi.x_0.value,color='r',dashes=[1,0.5])
        
        s.lorentzians_with_Bkg= s.lorentzians + custom_model(lambda x: s.Bkg_func(x))()
            
        s.set_modes(modes0_lor=s.modes0_lor,modes1_lor=s.modes1_lor,modes2_lor=s.modes2_lor)
        s.lList  = sorted(s.lList)
        s.lcolor = s._org_lcolor
        s.lmarker = s._org_lmarker
        
        
        return
            

#%% Fitter class
    #%%% mixed modes fit
class mixed_modes_Fit():
    def __init__(fitter,star=None,modes1=None,modes1_sig=None,np1=None
                ,pi_modes_init=None,gamma_modes_init=None,eps_g_init=None,q_initial=None
                ,dpi=None,dnu=None,plot_corner=True #gamma_modes in sec, dpi in sec, modes1 is μHz, pi_modes_init in μHz
                ,MCMC_or_dynesty='MCMC',dynamic=False,fit_dpi=False): 
        if star:
            s=star
            
        else:
            if not eps_g_init:
                eps_g_init=(gamma_modes_init % dpi) /dpi
            
            s=Star(modes1=modes1,modes1_sig=modes1_sig,np1=np1
                   ,pi_modes=pi_modes_init,gamma_modes=gamma_modes_init,eps_g1=eps_g_init
                   ,q=q_initial
                   ,dpi=dpi,dnu=dnu)
            
        s.fitter=fitter
        fitter.s=s

        plt.figure()
        ax_mixmodes=plt.gca()
        
        num_orders=len(np.unique(s.np1))
        
        fitter.results      =np.zeros([num_orders,3])
        fitter.uncertainties=np.zeros([num_orders,3,2])
        
        fitter.maxiter_halley=30
        
        ndim = 3
        if fit_dpi:
            ndim=ndim*num_orders+1
            
        fitter.nwalkers=100
        fitter.nsteps=5000
        
        if not fit_dpi:
            fitter.logz          =[None]*num_orders
          
            fitter.samplers=[]
            s.mixed_modes_theo=[]
        else:
            fitter.data_for_each_order=[]
        
        fitter.result_sample = [None]*num_orders
        fitter.result_resample=[None]*num_orders
        fitter.weights       = [None]*num_orders
            
        
        for i_np,npi in enumerate(np.unique(s.np1)):#[0:2]):
        
            mask_n=s.np1==npi    #!!! put all in data for each order and get rid of redefinition below
            mixed_modes_obs_n = s.modes1[mask_n]
            mixed_modes_sig_n = s.modes1_sig[mask_n]
            
            ng=s.mean_gamma_modes_sec[i_np]//s.dpi
            ng_tol=int(abs(np.median(np.diff(s.mean_gamma_modes_sec))/s.dpi))
            ng = np.arange(ng-ng_tol,ng+ng_tol)
            
            if fit_dpi:
                fitter.data_for_each_order.append({
                        'mixed_modes_obs': mixed_modes_obs_n
                       ,'mixed_modes_sig': mixed_modes_sig_n
                       ,'ng': ng
                       ,'initial_guess_pimodes':s.pi_modes[i_np]})
            else:
                fitter.initial_guess_pi_epsg_q=[s.pi_modes[i_np],s.eps_g1[i_np],s.q[i_np]]
                
                log_prob=fitter.make_calc_L_mixed_modes(mixed_modes_obs_n,ng,mixed_modes_sig_n,MCMC_or_dynesty=MCMC_or_dynesty,fit_dpi=fit_dpi)
            
                if MCMC_or_dynesty=='MCMC':
    
                    fitter.sampler = emcee.EnsembleSampler(fitter.nwalkers, ndim, log_prob)
                    fitter.sampler.run_mcmc(fitter.p0, fitter.nsteps, progress=True)
                    
                    fitter.samplers.append(fitter.sampler)
                    fitter.result_sample[i_np]  = fitter.sampler.get_chain(discard=int(0.5*fitter.sampler.iteration), flat=True)

                    
                else:
                    log_likelihood,prior_transform=log_prob
                    
                    if not dynamic:
                        fitter.sampler = dynesty.NestedSampler(log_likelihood, prior_transform, ndim, nlive=1000)
                        fitter.sampler.run_nested()
                    else:
                        fitter.sampler = dynesty.DynamicNestedSampler(log_likelihood, prior_transform, ndim, bound='single')
                        fitter.sampler.run_nested(dlogz_init=0.05, nlive_init=500, nlive_batch=100)
                    
                    
                    results = fitter.sampler.results
                    fitter.result_sample[i_np]=results.samples
                
                    fitter.weights[i_np] = np.exp(results.logwt - results.logz[-1])
                    # fitter.result_resample[i_np] = dyfunc.resample_equal(results.samples, fitter.weights[i_np])
                    
                    fitter.logz[i_np] = results.logz[-1]
                    fitter.samplers.append(fitter.sampler)
                    
                print(npi,' done' )
            
        if fit_dpi:
            log_prob_func = fitter.make_calc_L_mixed_modes(fit_dpi=True)
        
            if MCMC_or_dynesty == 'dynesty':
                
                log_likelihood, prior_transform = log_prob_func
                
                fitter.sampler = dynesty.DynamicNestedSampler(log_likelihood, prior_transform, ndim, bound='single')#,pool=pool,queue_size=2)
                fitter.sampler.run_nested(dlogz_init=0.05, nlive_init=500,nlive_batch=100,print_progress=True)
                
                results = fitter.sampler.results
                
                fitter.full_sample= results.samples
                fitter.weights = np.exp(results.logwt - results.logz[-1])
                
        labels=["π-modes (s)", "ε_g", "q"]
        if fit_dpi:
            labels_tot=labels*num_orders+['ΔΠ (s)']
            labels=labels+['ΔΠ (s)']
            
            
        for i_np,npi in enumerate(np.unique(s.np1)):
            
            #!!! ###################### #!!!
            mask_n=s.np1==npi
            mixed_modes_obs_n = s.modes1[mask_n]
            mixed_modes_sig_n = s.modes1_sig[mask_n]
            
            ng=s.mean_gamma_modes_sec[i_np]//s.dpi
            ng_tol=int(abs(np.median(np.diff(s.mean_gamma_modes_sec))/s.dpi))
            ng = np.arange(ng-ng_tol,ng+ng_tol)
            #!!! ###################### #!!!
            
            if fit_dpi:
                indices = [i_np*3, i_np*3+1, i_np*3+2, -1]
                fitter.result_sample[i_np]=fitter.full_sample[:, indices]
                
            if plot_corner:
                if MCMC_or_dynesty=='MCMC':
                    fitter.weights[i_np]=None
                # corner.corner(fitter.result_sample[i_np],weights=fitter.weights[i_np],labels=labels
                #              ,quantiles=[0.16, 0.5, 0.84],show_titles=True,title_fmt=".3f")
                    
                # fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
                # for i in range(3):
                #     axes[i].plot(fitter.sampler.get_chain()[:, :, i], alpha=0.3)
                #     axes[i].set_ylabel(labels[i])
                    
            results,uncertainties=plot_cornerplot(fitter.result_sample[i_np].T,labels=labels,xi=None,labelcbar='',mask=None, levels=[0.68, 0.95]
                                   ,statistics_type='probability density',bins = 50,num_models_goodxi=25,median_or_mode='mode'
                                   ,kde_2D_plot=True,idx_for_circular_parameters=[1],plot=plot_corner)
            
            if fit_dpi:
                result_dpi_Ms=results[-1]
                uncertainty_dpi=uncertainties[-1]
                results=results[:-1]
                uncertainties=uncertainties[:,-1]
            else:
                result_dpi_Ms=s.dpi_Ms
                
            fitter.results[i_np,:]         = results.copy()
            fitter.uncertainties[i_np,:,:] = uncertainties.copy()
            
            gamma_modes=1/((fitter.results[i_np,1]+ng)*result_dpi_Ms)
            mixed_modes_theo,idx_matched_modes_theo, idx_matched_modes_obs=fitter.calc_mixed_modes(
                                                                        mixed_modes_obs_n,pi_mode=fitter.results[i_np,0]
                                                                       ,gamma_modes=gamma_modes,q=fitter.results[i_np,2]
                                                                       ,dpi_Ms=result_dpi_Ms
                                                                       ,ax=ax_mixmodes
                                                                       )
            s.mixed_modes_theo.append(mixed_modes_theo[idx_matched_modes_theo])
        
        # if fit_dpi:
        #     corner.corner(fitter.full_sample,labels=labels_tot,weight=fitter.weights
        #                  ,quantiles=[0.16, 0.5, 0.84],show_titles=True,title_fmt=".3f")
                
                
        s.pi_modes_init=fitter.results[:,0].copy()
        s.pi_modes     =fitter.results[:,0].copy()
        s.eps_g1       =fitter.results[:,1].copy()
        s.q            =fitter.results[:,2].copy()
        s.eps_p1_init  =s.pi_modes_init % s.dnu / s.dnu
        
        ng  = (1/(s.pi_modes_init*1e-6)) // s.dpi
        s.mean_gamma_modes_sec=(s.eps_g1+ng) * s.dpi
        
        s.pi_modes_uncertainties=s.fitter.uncertainties[:,0,:].copy()
        s.eps_g1_uncertainties  =s.fitter.uncertainties[:,1,:].copy()
        s.q_uncertainties       =s.fitter.uncertainties[:,2,:].copy()
        
        s.fitter_result_sample=fitter.result_sample
        s.fitter_weights = fitter.weights
        
        if fit_dpi:
            s.dpi=result_dpi_Ms*1e-6
            s.dpi_uncertainty=uncertainty_dpi*1e-6
        
    #%%% stats
    def calc_mixed_modes(fitter,mixed_modes_obs,pi_mode,gamma_modes,q,dpi_Ms,plot=True,ax=None):
        s=fitter.s
        
        if not isinstance(pi_mode,(np.ndarray,list)):
            pi_mode=np.array([pi_mode])
        
        
        # t0 = time.perf_counter()
        mixed_p_mode, mixed_g_modes = asympthotic_theory(pi_mode=pi_mode, gamma_modes=gamma_modes, q=q, dpi_Ms=dpi_Ms, dnu=s.dnu, interp_kind=s.interp_kind
                                                         ).couple_halley(maxiter=fitter.maxiter_halley)
        # t1 = time.perf_counter()
        
        mixed_modes_theo=np.sort(np.append(mixed_p_mode,mixed_g_modes))
        idx_matched_modes_obs,idx_matched_modes_theo=fitter.match_modes(mixed_modes_obs,mixed_modes_theo,printSkips=False)
        # t2 = time.perf_counter()
        
        # print(f" {t0-t1:.6f} s")
        # print(f" {t1-t2:.6f} s")
        # print("##########")
        if plot:
            if not ax:
                plt.figure()
                ax=plt.gca()
            ax.plot(mixed_modes_theo % s.dnu,mixed_modes_theo,"o",color='k',ms=0.5)
            ax.plot(mixed_modes_obs % s.dnu,mixed_modes_obs,"x",color='b')
            ax.plot(mixed_modes_obs[  idx_matched_modes_obs]  % s.dnu,mixed_modes_obs[ idx_matched_modes_obs],"+",color='b')
            
            ax.plot(mixed_modes_theo[idx_matched_modes_theo]  % s.dnu,mixed_modes_theo[idx_matched_modes_theo],"x",color='r',ms=4,zorder=100)
        
        return mixed_modes_theo,idx_matched_modes_theo, idx_matched_modes_obs
    
    def make_calc_L_mixed_modes(fitter, mixed_modes_obs=None,ng=None,mixed_modes_sig=None, MCMC_or_dynesty='MCMC',fit_dpi=False):
        s=fitter.s
        
        if not fit_dpi:
            def log_likelihood(pi_epsg_q):
                pi_mode, epsg, q = pi_epsg_q
                dpi_Ms=s.dpi_Ms
                
                epsg = epsg % 1.0
                
                gamma_modes=1/((epsg+ng)*dpi_Ms)
                
                mixed_modes_theo,idx_matched_modes_theo, idx_matched_modes_obs=fitter.calc_mixed_modes(
                                                    mixed_modes_obs,pi_mode,gamma_modes,q,dpi_Ms,plot=False)
                
                X=np.sum(((mixed_modes_theo[idx_matched_modes_theo]- mixed_modes_obs[idx_matched_modes_obs])
                                      /mixed_modes_sig[idx_matched_modes_obs])**2)
                
                return -0.5*X
                # return X
                
        else:
            
            def log_likelihood(parameters):
                # parameters = [pi_0,eps_0,q_0, ..., pi_n,eps_n,q_n,dpi]
                dpi_Ms = parameters[-1]  * 1e-6
                
                logL = 0
                for n in range(len(np.unique(s.np1))):
                    pi_mode_n = parameters[n*3]
                    epsg_n    = parameters[n*3 + 1] % 1.0
                    q_n       = parameters[n*3 + 2]
                    
                    data_n = fitter.data_for_each_order[n]
                    gamma_modes_n = 1 / ((epsg_n + data_n['ng']) * dpi_Ms)
                    
                    mixed_modes_theo_n,idx_matched_modes_theo_n, idx_matched_modes_obs_n = fitter.calc_mixed_modes(
                                                        data_n['mixed_modes_obs'],pi_mode_n,gamma_modes_n,q_n, dpi_Ms,  plot=False)
                    
                    X=np.sum(((mixed_modes_theo_n[idx_matched_modes_theo_n]- data_n['mixed_modes_obs'][idx_matched_modes_obs_n])
                                                  /data_n['mixed_modes_sig'][idx_matched_modes_obs_n])**2)
                    
                    logL += -0.5*X
        
                return logL
        
        epsg_minmax = [fitter.initial_guess_pi_epsg_q[1]-0.5,fitter.initial_guess_pi_epsg_q[1]+0.5]
        q_minmax    = [0.,1]
        
        if not fit_dpi:    
            pi_minmax   = [fitter.initial_guess_pi_epsg_q[0] - 0.25 * s.dnu
                         , fitter.initial_guess_pi_epsg_q[0] + 0.25 * s.dnu]
            
            if MCMC_or_dynesty=='MCMC':
                def log_prior(pi_epsg_q):
                    pi_mode, epsg, q = pi_epsg_q
                
                    if not (pi_minmax[0]   < pi_mode < pi_minmax[1]):
                        return -np.inf
                    if not (epsg_minmax[0] < epsg    < epsg_minmax[1]):
                        return -np.inf
                    if not (q_minmax[0]     < q       < q_minmax[1]):
                        return -np.inf
                
                    return 0.0
                
                samples=[]
                for meani,minmaxi in zip(fitter.initial_guess_pi_epsg_q,[pi_minmax,epsg_minmax,q_minmax]):
                    
                    stddev=0.1 * (minmaxi[1] - minmaxi[0])
                    samples.append(truncnorm.rvs((minmaxi[0] - meani) / stddev
                                             ,(minmaxi[1] - meani) / stddev
                                             ,loc=meani, scale=stddev, size=fitter.nwalkers))
                fitter.p0=np.array(samples).T
                
                def log_probability(theta):
                    lp = log_prior(theta)
                    if not np.isfinite(lp):
                        return -np.inf
                    return lp + log_likelihood(theta)
                
                return log_probability
            
            else:
                def prior_transform(u):
                    v = np.array(u)
                    
                    v[0] = u[0] * (pi_minmax[1]  -pi_minmax[0])  + pi_minmax[0] 
                    v[1] = u[1] * (epsg_minmax[1]-epsg_minmax[0])+ epsg_minmax[0]      
                    v[2] = u[2] * (q_minmax[1]   -q_minmax[0])   + q_minmax[0]      
                    return v
                
                return log_likelihood,prior_transform
        
        else:
            
            def prior_transform(u):
                v = np.zeros_like(u)
    
                for n in range(len(np.unique(s.np1))):
                    
                    init_pi = fitter.data_for_each_order[n]['initial_guess_pimodes']
                    pi_minmax_n   = [init_pi - 0.1 * s.dnu
                                    ,init_pi + 0.1 * s.dnu]
                    
                    v[n*3]     = u[n*3]      * (pi_minmax_n[1] -pi_minmax_n[0])  + pi_minmax_n[0] 
                    v[n*3 + 1] = u[n*3 + 1]  * (epsg_minmax[1] -epsg_minmax[0])  + epsg_minmax[0]      
                    v[n*3 + 2] = u[n*3 + 2]  * (q_minmax[1]    -q_minmax[0])     + q_minmax[0]      
                    
                v[-1] = u[-1] * (0.02 * s.dpi) + (0.99 * s.dpi)
                return v
        
            return log_likelihood, prior_transform
    
    
    
    #%%% match modes
    
    # def match_modes_sliding_window(fitter,modes_obs, modes_theo, modes_sig):

    #     n_obs  = len(modes_obs)
    #     n_theo = len(modes_theo)
        
    #     if n_theo < n_obs:
    #         raise ValueError("too less theo modes")
    
    #     idx_best = 0
    #     error_min = np.inf
    
    #     for i in range(n_obs - n_theo + 1):
    #         diff = modes_obs - modes_theo[i : i+n_obs]
    #         error = np.sum((diff / modes_sig)**2)
    
    #         if error < error_min:
    #             error_min = error
    #             idx_best = i
    
    #     return modes_theo[idx_best : idx_best + n_obs]
    
    def match_modes(fitter,modes_obs,modes_theo,gap_penalty=None,printSkips=True): #dynamic time warping (DTW)

        n_obs = len(modes_obs)
        n_theo = len(modes_theo)
        
        if not gap_penalty:
            gap_penalty=(0.5*np.median(np.diff(modes_obs)))**2
            
        dp = np.ones((n_obs + 1, n_theo + 1))*np.inf
        dp[0, :] = 0   
        dp[:, 0] = 2*gap_penalty *np.inf
        
        diff = modes_obs[:, None] - modes_theo[None, :]
        cost_matrix = diff**2
        
        for i in range(1, n_obs + 1):
            for j in range(1, n_theo + 1):
                dp[i, j] = min(dp[i-1, j-1] + cost_matrix[i-1,j-1],  
                               dp[i-1, j  ] + 10*gap_penalty*np.inf,  
                               dp[i  , j-1] + gap_penalty)
        
        i_obs, i_theo = n_obs, n_theo
        alignment = []
        idx_matched_modes_obs=[]
        idx_matched_modes_theo=[]
        while i_obs > 0 or i_theo > 0:
            
            if i_obs == 0:
                alignment.append((None, modes_theo[i_theo-1]))
                i_theo -= 1
                continue
        
            elif i_theo == 0:
                alignment.append((modes_obs[i_obs-1], None))
                i_obs -= 1
                continue
        
            elif np.isclose(dp[i_obs, i_theo], dp[i_obs-1, i_theo-1] + cost_matrix[i_obs-1, i_theo-1]):
                alignment.append((modes_obs[i_obs-1], modes_theo[i_theo-1]))
                idx_matched_modes_obs.append(i_obs-1)
                idx_matched_modes_theo.append(i_theo-1)
                i_obs -= 1
                i_theo -= 1
                
            elif np.isclose(dp[i_obs, i_theo],dp[i_obs, i_theo-1] + gap_penalty):
                alignment.append((None, modes_theo[i_theo-1]))  
                if not i_obs==n_obs:
                    if printSkips:
                        print()
                        print('skipped theoretical mode!!!')
                        print()
                i_theo -= 1
                
            else:
                alignment.append((modes_obs[i_obs-1],None))  
                if printSkips:
                    print()
                    print('skipped observed mode!!!',modes_obs[i_obs-1])
                    print()
                i_obs -= 1
                
        alignment.reverse()
        idx_matched_modes_obs.reverse()
        idx_matched_modes_theo.reverse()
        
        return idx_matched_modes_obs,idx_matched_modes_theo

        
#%%
########################################################################################################














