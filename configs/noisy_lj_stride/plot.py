"""Plot the measured sampling comparison; run with the analysis dependencies."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).parent
result=json.loads((root/'comparison.json').read_text())
rows={(row['variant'],row['model']):row for row in result['summary']}
plt.rcParams.update({'font.size':16,'axes.titlesize':19,'axes.labelsize':17,'xtick.labelsize':16,'ytick.labelsize':16})
fig,axes=plt.subplots(2,2,figsize=(20,13))
settings=[
    (15,['dense_head','dense_spread','stride2_spread','stride5_spread'],['Head\nEvery frame','Spread\nEvery frame','Spread\nEvery 2nd','Spread\nEvery 5th']),
    (3,['published_head_original','published_mst_original','dense_spread_original','dense_short_original','stride2_short_original'],['One-step\n4 inputs','MST\n4 inputs','Spread\n4 inputs','Spread\n2 inputs','Every 2nd\n2 inputs']),
]
for row,(anchor,variants,labels) in enumerate(settings):
    for column,metric in enumerate(['test_r2','relative_mse']):
        ax=axes[row,column]
        for index,(model,color,hatch) in enumerate([('mlp','#0072b2',''),('edge_mlp','#d55e00','//')]):
            values=[rows[variant,model][f'{metric}_mean'] for variant in variants]
            sd=[rows[variant,model][f'{metric}_sd'] for variant in variants]
            drawn=np.maximum(values,0)
            lower=drawn-np.maximum(0,np.array(values)-sd)
            upper=np.maximum(0,np.array(values)+sd)-drawn
            bars=ax.bar(np.arange(len(variants))+(index-.5)*.36,drawn,.36,color=color,hatch=hatch,
                        edgecolor='white',label=model,yerr=[lower,upper],capsize=4)
            for bar,value,error in zip(bars,values,upper):
                label_offset=.012
                if metric=='relative_mse' and index==0:
                    label_offset+=.015 if row==0 else .025
                ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+error+label_offset,f'{value:.3f}',ha='center',va='bottom',fontsize=16)
        ax.set_xticks(range(len(variants)),labels)
        ax.set_title(f'Observed through frame {anchor}; predict to frame {anchor+100}',pad=14)
        ax.set_ylabel('Test Poisson R² ↑' if metric=='test_r2' else 'Relative position MSE ↓')
        maximum=max(rows[variant,model]['relative_mse_mean']+rows[variant,model]['relative_mse_sd'] for variant in variants for model in ['mlp','edge_mlp'])
        ax.set_ylim(0,1 if metric=='test_r2' else maximum*(1.5 if row==0 else 1.25))
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
        if row==0 and column==0:ax.legend(loc='lower left')
fig.suptitle('Noisy LJ: MLP sampling experiments',fontsize=24,y=.98)
fig.text(.5,.025,'Same 50 training / 50 validation / 70 test networks · Mean ± sample SD over 3 training seeds\n'
         'Compare within each row: later observed frames change the task. Negative R² is drawn at 0; labels show actual values.',ha='center',fontsize=16)
fig.subplots_adjust(left=.07,right=.98,top=.91,bottom=.14,hspace=.4,wspace=.22)
fig.savefig(root/'comparison.png',dpi=160)
fig.savefig(root/'comparison.pdf')
print('Saved comparison.png and comparison.pdf')
