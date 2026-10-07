"""Plot recorded aggregate validation evidence; no local model artifacts required.

The values are transcribed from the documented Mac runs, not new model results.
See docs/visual_results_guide.md for provenance and rounding details.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    output = Path(__file__).resolve().parents[1] / "results/figures"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold'})
    fig,axs=plt.subplots(2,2,figsize=(16,12))
    fig.subplots_adjust(left=.12,right=.97,top=.86,bottom=.19,wspace=.33,hspace=.65)
    fig.suptitle('PaymentGuard | From model performance to individual scores',x=.06,ha='left',y=.98,fontsize=23,fontweight='bold',color='#19364c')
    fig.text(.06,.935,'PREVIEW  •  Synthetic AMLNet data  •  Validation only  •  1,680 reviews out of 167,990 transactions',fontsize=12,color='#526572')
    teal='#008578'; orange='#cb7537'; navy='#19364c'
    a=axs[0,0]
    labels=['Amount ranking','Full log model','Log without category','Log without type\n+ category'];v=[109,196,130,113]
    b=a.barh(labels,v,color=['#9aaab5',teal,orange,'#9aaab5']);a.invert_yaxis();a.bar_label(b,labels=[f'{n} / 202' for n in v],padding=6);a.set_xlim(0,245);a.set_xlabel('Positive labels caught');a.set_title('1  Category removal reduced detection',loc='left',pad=20)
    a.text(0,-.27,'Full log model: 196 caught + 1,484 false alarms\n97.0% recall; 11.7% precision at this review budget.',transform=a.transAxes,fontsize=11,color=navy)
    a=axs[0,1];cats=['Housing','Other','Recreation','Shell Company','Education'];y=np.arange(5)
    a.barh(y-.19,[22,1517,69,68,4],.36,color=teal,label='Full log model');a.barh(y+.19,[1519,109,12,40,0],.36,color=orange,label='Log without category');a.set_yticks(y,cats);a.invert_yaxis();a.set_xlim(0,1900);a.set_xlabel('Transactions sent for review');a.set_title('2  Housing absorbed the review budget',loc='left',pad=20);a.legend(fontsize=10,loc='lower right');a.text(1530,.19,'1,519',va='center',color=orange,fontweight='bold');a.text(1530,.81,'1,517',va='center',color=teal,fontweight='bold')
    a.text(0,-.27,'Without category: 90.4% of reviews went to Housing.\nAll 1,519 were labelled non-fraud. Four zero-review categories omitted.',transform=a.transAxes,fontsize=10,color=navy)
    for ax,amount,balance,end,rank,prob,ref in [(axs[1,0],5.806329,-.174908,-2.877067454324905,1680,'5.329891%',671963),(axs[1,1],5.191004,-.047504,-2.877071317021401,1681,'5.329872%',706036)]:
        intercept=-10.596675731943682; transfer=3.694723;other=end-intercept-amount-transfer-balance
        vals=[intercept,amount,transfer,balance,other];total=0
        for i,val in enumerate(vals):
            start=0 if i==0 else total
            ax.bar(i,val,bottom=start,color=navy if i==0 else (teal if val>=0 else orange),width=.65)
            total=start+val
            ax.text(i,max(start,total)+.28,f'{val:+.3f}',ha='center',fontsize=10)
            ax.plot([i+.325,i+.675],[total,total],color='#aab7bf',lw=1)
        ax.bar(5,end,color=navy,width=.65);ax.text(5,.28,f'{end:.3f}',ha='center',fontsize=10)
        ax.axhline(0,color='#81909b',lw=.7);ax.set_ylim(-12,2);ax.set_xticks(range(6),['Intercept','Log\namount','TRANSFER','Origin\nbalance','Other\nfeatures','Final\nscore']);ax.set_ylabel('Log-odds contribution (not probability)')
        ax.set_title(f'3{"a" if rank==1680 else "b"}  {"Selected" if rank==1680 else "Not selected"}: rank {rank:,}',loc='left',pad=20)
        ax.text(0,-.29,f'Model probability: {prob}  |  Actual label: non-fraud\nSource row {ref}; same model without category.',transform=ax.transAxes,fontsize=11,color=navy)
    fig.text(.06,.045,'Reading the bottom panels: amount and transfer type raise both scores; other inputs partly offset them.\nThe two probabilities are almost identical. The fixed review capacity admits rank 1,680 and excludes rank 1,681.',fontsize=12,color=navy)
    fig.text(.06,.009,'Source: recorded ablation metrics and verified local score diagnostics. Two illustrative cases, not causal explanations. Contributions rounded; other features grouped.',fontsize=9,color='#526572')

    f = fig
    for t in list(f.texts): t.remove()
    f.set_size_inches(16,7.5)
    for a in axs[1]: a.set_visible(False)
    for a,pos in zip(axs[0],[[.15,.31,.34,.43],[.63,.31,.34,.43]]): a.set_position(pos)
    f.text(.05,.94,'PaymentGuard | What changed when category was removed?',fontsize=22,fontweight='bold',color='#19364c')
    f.text(.05,.885,'Synthetic AMLNet data  |  Validation only  |  Same budget: 1,680 reviews  |  202 positive labels',fontsize=12,color='#526572')
    f.text(.05,.065,'Takeaway: the full log model caught more positives. After removing category and retraining,\nmost reviews went to Housing transactions labelled non-fraud.',fontsize=13,color='#19364c')
    f.text(.05,.018,'Source: recorded validation metrics and category diagnostics. These results do not establish real-world performance.',fontsize=10,color='#526572')
    f.savefig(output / 'paymentguard_model_overview.png',dpi=160,facecolor='white')
    for t in list(f.texts): t.remove()
    for a in axs[0]: a.set_visible(False)
    for a,pos in zip(axs[1],[[.10,.31,.38,.43],[.60,.31,.38,.43]]):
        a.set_visible(True);a.set_position(pos)
    axs[1,0].set_title('Selected for review | Rank 1,680',loc='left',pad=20)
    axs[1,1].set_title('Not selected | Rank 1,681',loc='left',pad=20)
    f.text(.05,.94,'PaymentGuard | How two Housing scores were calculated',fontsize=22,fontweight='bold',color='#19364c')
    f.text(.05,.885,'Log-amount model without category  |  Two validation examples  |  Both labelled non-fraud',fontsize=12,color='#526572')
    f.text(.05,.815,'Read left to right: starting score + feature contributions = final score. Teal raises it; orange lowers it.',fontsize=12,color='#19364c')
    f.text(.05,.065,'Takeaway: different contributions produced almost identical probabilities.\nThe fixed review budget admitted rank 1,680 and excluded rank 1,681.',fontsize=13,color='#19364c')
    f.text(.05,.018,'Source: verified local prediction reconstruction. Contributions rounded; other features grouped. Illustrative model explanations, not causal effects.',fontsize=10,color='#526572')
    f.savefig(output / 'paymentguard_score_explanation.png',dpi=160,facecolor='white')

    plt.close(f)


if __name__ == "__main__":
    main()
