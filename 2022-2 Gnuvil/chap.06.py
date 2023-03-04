#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pip install wget


# In[2]:


import wget
url = 'https://bit.ly/fruits_300_data'
wget.download(url)


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# In[ ]:





# In[ ]:


fruits = np.load('fruits_300_data')


# In[ ]:





# In[ ]:


print(fruits.shape)


# In[ ]:


print(fruits[0,0, :])   


# In[ ]:





# In[ ]:


plt.imshow(fruits[0], cmap='gray')


# In[ ]:


plt.show()


# In[ ]:


plt.imshow(fruits[0], cmap='gray_r')


# In[ ]:


plt.show()


# In[ ]:


fig, axs = plt.subplots(1, 2)
axs[0].imshow(fruits[100], cmap='gray_r')
axs[1].imshow(fruits[200], cmap='gray_r')
plt.show()


# In[ ]:





# In[ ]:


apple = fruits[0:100].reshape(-1, 100*100)
pineapple = fruits[100:200].reshape(-1, 100*100)
banana = fruits[200:300].reshape(-1, 100*100)


# In[ ]:


print(apple.mean(axos=1))


# In[ ]:


plt.hist(np.mean(apple, axis = 1), alpha=0.8)
plt.hist(np.mean(pineapple, axis=1), alpha=0.8)
plt.hist(np.mean(banana, axis=1), alpha=0.8)
plt.legend(['apple', 'pineapple', 'banana'])
plt.show()


# In[ ]:


fig, axs = plt.subplots(1, 3, figsize=(20,5))
axs[0].bar(range(10000), np.mean(apple, axis=0))
axs[1].bar(range(10000), np.mean(pineapple, axis=0))
axs[2].bar(range(10000), np.mean(banana, axis=0))
plt.show()


# In[ ]:


apple_mean = np.mean(apple, axis = 0).reshape(100, 100)
pineapple_mean = np.mean(pineapple, axis=0).reshape(100, 100)
banana_mean = np.mean(banana, axis = 0).reshape(100, 100)
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
axs[0].imshow(apple_mean, cmap='gray_r')
axs[1].imshow(pineapple_mean, cmap='gray_r')
axs[2].imshow(banana_mean, cmap='gray_r')
plt.show()


# In[ ]:





# In[ ]:


abs_diff = np.abs(fruits - apple_mean)
abs_mean = np.mean(abs_diff, axis = (1, 2))
print(abs_mean.shape)


# In[ ]:


apple_index = np.argsort(abs_mean)[:100]
fig, axs = plt.subplots(10, 10, figsize = (10, 10))
for i in range(10):
    for j in range(10):
        axs[i, j].imshow(fruits[apple_index[i*10 + j]], cmap='gray_r')
        axs[i, j].axis('off')
plt.show()


# In[ ]:





# In[ ]:


#kMeans


# In[ ]:


fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)


# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


km = KMeans(n_cluster=3, random_state=42)
km.fit(fruits_2d)
print(km.labels_)


# In[ ]:


print(np.unique(km.labels_, return_counts=True))


# In[ ]:


def draw_fruits(arr, ratio=1):
    n = len(arr)
    rows = int(np.ceil(n/10))
    cols = n if rows < 2 else 10
    fig, axs = plt.subplots(rows, cols, figsize=(cols*ratio, rows*ratio), squeeze=False)
    
    for i in range(rows):
        for j in range(cols):
            if i*10 + j < n:
                axs[i, j].imshow(arr[i*10 + j], cmap = 'gray_r')
            axs[i, j].axis('off')
plt.show()


# In[ ]:


draw_fruits(fruits[km.labels_==0])


# In[ ]:


draw_fruits(fruits[km.labels_==1])


# In[ ]:


draw_fruits(fruits[km.labels_==2])


# In[ ]:





# In[ ]:


draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), ratio=3)


# In[ ]:


print(km.transform(fruits_2d[100:101]))


# In[ ]:


print(km.predict(fruits_2d[100:101]))


# In[ ]:


draw_fruits(fruits[100:101])


# In[ ]:


print(km.n_iter_) #알고리즘 반복 횟수


# In[ ]:





# In[ ]:


#최적의 k 찾기


# In[ ]:


inertia = []
for k in range(2, 7):
    km = KMeans(n_clusters = k, random_state = 42)
    km.fit(fruits_2d)
    inertia.append(km.inertia_)
plt.plot(range(2, 7), inertia)
plt.xlabel('k')
plt.ylabel('inertia')
plt.show()


# In[ ]:





# In[ ]:


# 주성분 분석 PCA


# In[ ]:


fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)


# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=50)
pca.fit(fruits_2d)
print(pca.components_.shape)


# In[ ]:


draw_fruits(pca.components_.reshape(-1, 100, 100))


# In[ ]:


print(fruits_2d.shape) # 차원 줄이기


# In[ ]:


fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)


# In[ ]:


fruits_inverse = pca.inverse_transform(fruits_pca) #복원
print(fruits_inverse.shape)


# In[ ]:


fruits_reconstruct = fruits_inverse.reshape(-1, 100, 100)
for start in [0, 100, 200]:
    draw_fruits(fruits_reconstruct[start:start+100])
    print("\n")


# In[ ]:


print(np.sum(pca.explained_variance_ratio_)) #설명된 분산


# In[ ]:


plt.plot(pca.explained_variance_ratio_)
plt.show()


# In[ ]:





# In[ ]:


#다른 알고리즘과 함께 사용하기


# In[ ]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


# In[ ]:


target = np.array([0]*100 + [1]*100 + [2]*100)


# In[ ]:


from sklearn.model_selection import cross_validate
scores = cross(validate(lr, fruits_2d, target))
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))


# In[ ]:


scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))


# In[ ]:





# In[ ]:


pca = PCA(n_components=0.5)
pca.fit(fruits_2d)


# In[ ]:


print(pca.n_components_)


# In[ ]:


fruits_pca = pca.transform(fruits_2d) #원본 데이터 변환
print(fruits_pca.shape)


# In[ ]:


scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))


# In[ ]:


from sklearn.cluster import KMeans
km = KMeans(n_clusters = 3, random_state=42)
km.fit(fruits_pca)
print(np.unique(km.labels_, return_counts=True))


# In[ ]:


for label in range(0, 3):
    draw_fruits(fruits[km.labels_ == label])
    print("\n")


# In[ ]:


for label in range(0, 3):
    data = fruits_pca[km.labels_ == label]
    plt.scatter(data[:, 0], data[:,1])
plt.legend(['apple', 'pineapple', 'banana'])
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




