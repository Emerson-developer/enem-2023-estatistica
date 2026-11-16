# Trabalho de Probabilidade e Estatística para IA — ENEM 2023 (microdados INEP)
import numpy as np, pandas as pd, json, math, sys, os
from scipy import stats
SRC = '/mnt/user-data/uploads/microdados_enem_2023/trabalho_estatistica/enem_sel.csv.gz'  # colunas extraídas de MICRODADOS_ENEM_2023.csv
raw = pd.read_csv(SRC, sep=';', encoding='latin-1', dtype=str)
# POPULAÇÃO: inscritos presentes nas 4 provas objetivas (os dois dias de prova)
pres = raw[(raw.TP_PRESENCA_CN=='1')&(raw.TP_PRESENCA_CH=='1')&(raw.TP_PRESENCA_LC=='1')&(raw.TP_PRESENCA_MT=='1')].copy()
for c in ['NU_NOTA_MT','NU_NOTA_LC','NU_NOTA_CH','NU_NOTA_CN','NU_NOTA_REDACAO']: pres[c]=pd.to_numeric(pres[c])
pres['escola']=pres.TP_ESCOLA.map({'1':'Não informado','2':'Pública','3':'Privada'})
pres['sexo']=pres.TP_SEXO.map({'F':'Feminino','M':'Masculino'})
pres['situacao']=pres.TP_ST_CONCLUSAO.map({'1':'Já concluiu','2':'Concluinte 2023','3':'Treineiro','4':'Não concluiu e não cursa'})
pres['internet']=pres.Q025.map({'A':'Não','B':'Sim'})
REG={'Norte':'AC AM AP PA RO RR TO','Nordeste':'AL BA CE MA PB PE PI RN SE','Sudeste':'ES MG RJ SP','Sul':'PR RS SC','Centro-Oeste':'DF GO MS MT'}
uf2r={u:r for r,us in REG.items() for u in us.split()}
pres['regiao']=pres.SG_UF_PROVA.map(uf2r)
def faixa(q):  # salário mínimo 2023 = R$ 1.320
    if q in 'AB': return 'Até 1 SM'
    if q in 'CD': return '1 a 2 SM'
    if q in 'EFGH': return '2 a 5 SM'
    if q in 'IJKLM': return '5 a 10 SM'
    return 'Mais de 10 SM'
pres['renda']=pres.Q006.map(faixa)
pop=pres.reset_index(drop=True); N=len(pop)
# TAMANHO AMOSTRAL (fórmula da calculadora USP/Bauru, população finita)
conf=0.95; e=0.03; p0=0.5
z=stats.norm.ppf(1-(1-conf)/2); n0=z**2*p0*(1-p0)/e**2; n=math.ceil(N*n0/(n0+N-1))
# AMOSTRA ALEATÓRIA SIMPLES sem reposição, semente fixa 2026
rng=np.random.default_rng(2026); idx=np.sort(rng.choice(N,size=n,replace=False))
am=pop.iloc[idx].copy(); am.insert(0,'id_pop',idx)
QV={'NU_NOTA_MT':'Matemática','NU_NOTA_LC':'Linguagens','NU_NOTA_CH':'Ciências Humanas','NU_NOTA_CN':'Ciências da Natureza','NU_NOTA_REDACAO':'Redação'}
CV={'escola':'Tipo de escola','sexo':'Sexo','renda':'Renda familiar','regiao':'Região','internet':'Internet em casa','situacao':'Situação no ensino médio'}
ORDEM={'renda':['Até 1 SM','1 a 2 SM','2 a 5 SM','5 a 10 SM','Mais de 10 SM']}
BINS=np.arange(0,1041,40)
GR=['Todos','Pública','Privada','Não informado']
def sub(d,g): return d if g=='Todos' else d[d.escola==g]
def resumo(d):
    rows=[]
    for g in GR:
        s=sub(d,g)
        for v,l in QV.items():
            x=s[v]; rows.append(dict(escola=g,variavel=v,rotulo=l,n=int(len(x)),media=round(x.mean(),2),mediana=round(float(x.median()),1),dp=round(x.std(),2),minimo=float(x.min()),q1=round(float(x.quantile(.25)),1),q3=round(float(x.quantile(.75)),1),maximo=float(x.max()),cv=round(x.std()/x.mean(),3)))
    return rows
def freq(d):
    rows=[]
    for g in GR:
        s=sub(d,g)
        for v,l in CV.items():
            vc=s[v].value_counts()
            for k,c in vc.items(): rows.append(dict(escola=g,variavel=v,rotulo=l,categoria=str(k),n=int(c),prop=round(c/len(s),4)))
    return rows
def hist(d):
    rows=[]
    for g in GR:
        s=sub(d,g)
        for v in QV:
            c,_=np.histogram(s[v].clip(0,1039.9),BINS)
            for i in range(len(c)): rows.append(dict(escola=g,variavel=v,ini=float(BINS[i]),fim=float(BINS[i+1]),n=int(c[i]),prop=round(c[i]/len(s),5)))
    return rows
# ---------- INFERÊNCIA (só com a amostra) ----------
alpha=0.05
def pv_of(stat,alt,dist):
    return 2*dist.sf(abs(stat)) if alt=='bilateral' else dist.sf(stat) if alt=='maior' else dist.cdf(stat)
def teste_prop(nome,var,evento,ph0,alt,h0,h1,just):
    x=int((am[var]==evento).sum()); ph=x/n; ep=math.sqrt(ph*(1-ph)/n); zc=stats.norm.ppf(1-alpha/2)
    zt=(ph-ph0)/math.sqrt(ph0*(1-ph0)/n); pv=pv_of(zt,alt,stats.norm); pp=(pop[var]==evento).mean()
    return dict(id=nome,tipo='Proporção',variavel=var,rotulo=f"{CV[var]} = {evento}",h0=h0,h1=h1,alternativa=alt,valor_h0=ph0,alpha=alpha,n=n,sucessos=x,
        estimativa=round(ph,4),ep=round(ep,5),ic_inf=round(ph-zc*ep,4),ic_sup=round(ph+zc*ep,4),estatistica='z',valor_estatistica=round(zt,3),
        valor_critico=round((stats.norm.ppf(1-alpha)*(-1 if alt=='menor' else 1)) if alt!='bilateral' else zc,3),p_valor=float(f"{pv:.3g}"),
        decisao='Rejeita H0' if pv<alpha else 'Não rejeita H0',parametro_pop=round(pp,4),
        condicoes=f"n·p0 = {n*ph0:.0f} ≥ 10 e n·(1−p0) = {n*(1-ph0):.0f} ≥ 10; AAS com n/N = {n/N:.3%} < 5% (sem correção para população finita)",justificativa=just)
def teste_media(nome,var,mu0,alt,h0,h1,just):
    x=am[var]; m=x.mean(); s=x.std(); ep=s/math.sqrt(n); df=n-1; tc=stats.t.ppf(1-alpha/2,df)
    tt=(m-mu0)/ep; pv=pv_of(tt,alt,stats.t(df)); mp=pop[var].mean()
    return dict(id=nome,tipo='Média',variavel=var,rotulo=f"Nota de {QV[var]}",h0=h0,h1=h1,alternativa=alt,valor_h0=mu0,alpha=alpha,n=n,sucessos=None,
        estimativa=round(m,2),ep=round(ep,3),ic_inf=round(m-tc*ep,2),ic_sup=round(m+tc*ep,2),estatistica=f't (gl = {df})',valor_estatistica=round(tt,3),
        valor_critico=round((stats.t.ppf(1-alpha,df)*(-1 if alt=='menor' else 1)) if alt!='bilateral' else tc,3),p_valor=float(f"{pv:.3g}"),
        decisao='Rejeita H0' if pv<alpha else 'Não rejeita H0',parametro_pop=round(mp,2),
        condicoes=f"σ desconhecido → t de Student; n = {n} ≥ 30, pelo TCL x̄ é aproximadamente normal (assimetria amostral = {stats.skew(x):.2f}); AAS garante independência",justificativa=just)
T=[
 teste_prop('P1','internet','Não',0.10,'menor','H0: p ≥ 0,10','H1: p < 0,10','Menos de 10% dos participantes fazem o ENEM sem internet em casa?'),
 teste_prop('P2','sexo','Feminino',0.50,'maior','H0: p ≤ 0,50','H1: p > 0,50','As mulheres são maioria entre os participantes presentes?'),
 teste_media('M1','NU_NOTA_MT',500,'maior','H0: μ ≤ 500','H1: μ > 500','A escala do ENEM foi criada com média 500: a média de Matemática em 2023 está acima disso?'),
 teste_media('M2','NU_NOTA_CN',500,'menor','H0: μ ≥ 500','H1: μ < 500','Ciências da Natureza costuma ter as menores notas: a média ficou abaixo de 500?'),
]
for t in T:
    v=t['parametro_pop']; h=t['valor_h0']; a=t['alternativa']
    sd=math.sqrt(v*(1-v)) if t['tipo']=='Proporção' else pop[t['variavel']].std()
    d=(v-h)/(sd/math.sqrt(n)); zc=stats.norm.ppf(1-alpha/2) if a=='bilateral' else stats.norm.ppf(1-alpha)
    t['poder_no_valor_real']=round(stats.norm.sf(zc-d)+stats.norm.cdf(-zc-d) if a=='bilateral' else stats.norm.sf(zc-d) if a=='maior' else stats.norm.cdf(-zc-d),3)
    t['n_para_poder_80']=math.ceil(((zc+stats.norm.ppf(.8))*sd/abs(v-h))**2) if v!=h else None
    t['h0_verdadeira_pop']='Sim' if ((a=='maior' and v<=h) or (a=='menor' and v>=h) or (a=='bilateral' and v==h)) else 'Não'
    t['ic_contem_pop']='Sim' if t['ic_inf']<=v<=t['ic_sup'] else 'Não'
# ---------- SIMULAÇÃO (1000 AAS de tamanho n) ----------
rs=np.random.default_rng(7); sim=[]; cob={'P1':0,'M1':0}
a1=(pop.internet=='Não').values; a2=pop.NU_NOTA_MT.values; pp1=a1.mean(); pm1=a2.mean(); tc=stats.t.ppf(.975,n-1); R=1000
for r in range(R):
    ii=rs.choice(N,n,replace=False)
    ph=a1[ii].mean(); ep=math.sqrt(ph*(1-ph)/n); lo,hi=ph-1.96*ep,ph+1.96*ep
    xm=a2[ii].mean(); s=a2[ii].std(ddof=1)/math.sqrt(n); lo2,hi2=xm-tc*s,xm+tc*s
    c1=lo<=pp1<=hi; c2=lo2<=pm1<=hi2; cob['P1']+=c1; cob['M1']+=c2
    if r<100:
        sim.append(dict(amostra=r+1,parametro='P1',estimativa=round(ph,4),ic_inf=round(lo,4),ic_sup=round(hi,4),contem='Sim' if c1 else 'Não'))
        sim.append(dict(amostra=r+1,parametro='M1',estimativa=round(xm,2),ic_inf=round(lo2,2),ic_sup=round(hi2,2),contem='Sim' if c2 else 'Não'))
cobertura=[dict(parametro='P1',rotulo='Proporção sem internet em casa',replicacoes=R,cobertura=cob['P1']/R,valor_pop=round(pp1,4)),
           dict(parametro='M1',rotulo='Média de Matemática',replicacoes=R,cobertura=cob['M1']/R,valor_pop=round(pm1,2))]
plano=[dict(item='Inscritos na base',valor=len(raw)),dict(item='População (N)',valor=N),dict(item='Inscritos excluídos (faltaram ou foram eliminados)',valor=len(raw)-N),
       dict(item='Nível de confiança',valor=conf),dict(item='z',valor=round(z,4)),dict(item='Margem de erro',valor=e),dict(item='Proporção p usada',valor=p0),
       dict(item='n0 (pop. infinita)',valor=round(n0,2)),dict(item='Tamanho da amostra (n)',valor=n),dict(item='Fração amostral n/N',valor=round(n/N,6)),dict(item='Semente aleatória',valor=2026)]
code=open(sys.argv[0]).read()
meta=[dict(label='analise_enem.py',value=code,type='code',lang='python'),dict(label='Entrada',value='MICRODADOS_ENEM_2023.csv (INEP), colunas selecionadas')]
os.makedirs('out',exist_ok=True)
def dump(name,rows): json.dump(dict(rows=rows,meta=meta),open(f'out/{name}.json','w'),ensure_ascii=False)
dump('pop_resumo',resumo(pop)); dump('pop_freq',freq(pop)); dump('pop_hist',hist(pop))
dump('amo_resumo',resumo(am)); dump('amo_freq',freq(am)); dump('amo_hist',hist(am))
dump('inferencia',T); dump('simulacao',sim); dump('cobertura',cobertura); dump('plano_amostral',plano)
am[['id_pop','SG_UF_PROVA','regiao','escola','sexo','renda','internet','situacao','NU_NOTA_MT','NU_NOTA_LC','NU_NOTA_CH','NU_NOTA_CN','NU_NOTA_REDACAO']].to_csv('out/amostra_aas_enem.csv',index=False)
for t in T: print({k:t[k] for k in ['id','sucessos','estimativa','ic_inf','ic_sup','valor_estatistica','p_valor','decisao','parametro_pop','ic_contem_pop','h0_verdadeira_pop','poder_no_valor_real','n_para_poder_80']})
print(cobertura, 'N',N,'n',n)
for k in ['escola','renda','regiao','situacao']: print(pop[k].value_counts(normalize=True).round(4).to_dict()); print(am[k].value_counts(normalize=True).round(4).to_dict())
print(pop.groupby('escola')[['NU_NOTA_MT','NU_NOTA_REDACAO']].mean().round(1))
