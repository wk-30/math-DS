import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import plotly.express as px

st.title('算数問題で行うデータサイエンス')

df = pd.read_csv('sansu-data.csv',skiprows=1)
#カラム削除 大問、中問、小問列を削除 サブ単元削除
df = df.drop(['s_q_num','part','m_q_no','s_q_no'],axis=1)

#欠損値の削除
df = df.dropna()

#入試問題ID「exam_name」を先頭３行から新たに作成し、一番左へ持ってくる。（作成前のカラムは削除）
exam= df['s_name'] + df['year'].astype(str) +"-" + df['time'].astype(str)
df.insert(0,"exam_name",exam)
#学校名、年度、回数のカラム削除 問題IDを作成後のため 
df = df.drop(['s_name','year','time'],axis=1)

#問題のカテゴリ変数を共通化「カテゴリー難度ー形式」「unit_diff_format」をmain_unitとdifficultyとproblem_formatを結合して、左から３列目へ持ってくる。
unit_diff_format= df['main_unit'] + "-" + df['difficulty'].astype(str) + "-" + df['problem_format'].astype(str) 
df.insert(2,"unit_diff_format",unit_diff_format)

#（作成前のカラムは削除）
df = df.drop(['main_unit','difficulty','problem_format'],axis=1)

#ダミー変数化
df = pd.get_dummies(df, drop_first=False,prefix='', prefix_sep='',columns=['Category', 'unit_diff_format'])

#配点【allocation】にカテゴリカル変数をかける　　グループバイで配点を合計し比率を計算
df.iloc[:, 2:] = df.iloc[:, 2:].mul(df.iloc[:, 1],axis=0)

#一つ一つの試験としてグループ化する
df_exam = df.groupby('exam_name').sum()

#わかりやすい表面的なデータフレームを見せる
st.write('データの全体像を確認してみましょう')

st.dataframe(df_exam)
#合計点で割る。
df_exam_nomalized = df_exam.iloc[:, 1:].div(df_exam.iloc[:, 0],axis=0)
# 標準化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

df_exam_scaled = scaler.fit_transform(df_exam_nomalized)


#データフレーム化
df_exam_scaled  = pd.DataFrame(df_exam_scaled, index=df_exam_nomalized.index, columns=df_exam_nomalized.columns)

st.write('樹形図：類似度が近いものから線で結んでいます')

from scipy.cluster.hierarchy import linkage
Z = linkage(df_exam_scaled,method="ward", metric="euclidean")

from scipy.cluster.hierarchy import dendrogram
fig2, ax2 = plt.subplots(figsize=(10,10), dpi=400)
ax2 = dendrogram(Z,
labels=list(df_exam_scaled.index),
orientation= 'right',

)

st.pyplot(fig2)


# # index=df_exam_nomalized.index,
# #データフレーム化
# df_exam_scaled  = pd.DataFrame(df_exam_scaled, columns=df_exam_nomalized.columns)
# df_exam_scaled

# #主成分分析
# from sklearn.decomposition import PCA
# pca = PCA(random_state=0)
# X_pc =  pca.fit_transform(df_exam_scaled)
# df_pca = pd.DataFrame(X_pc, index=df_exam_nomalized.index,columns=["PC{}".format(i + 1) for i in range(len(X_pc[0]))])
# print("主成分の数：　", pca.n_components_)
# print("保たれている情報：　", np.sum(pca.explained_variance_ratio_))
# display(df_pca)
