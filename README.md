# Public-vehicle-data-analysis
(LH) 공공주택 공유차량   적정 규모 산정

이 프로젝트는 하남시의 공공차량 이용 데이터를 분석하여 미래 수요를 예측하고 효율적인 차량 운영 방안을 도출하는 것을 목표로 합니다. 
시계열 분석 기법을 활용하여 데이터의 패턴을 파악하고, 이를 통해 정책 결정에 도움이 되는 인사이트를 제공하고자 합니다.

### 프로젝트 진행 예시

### : 머신러닝 하남시 시계열 예측

공공차량 이용현황 트렌드 분석을 위하여 객관적인 이용 현황과 세대수 데이터를 이용하여 화성시 데이터 분석기반 이용예측모델을 만들었다.

생성된 예측모델로 하남시 시계열 예측을 통해 공공차량 이용 추이를 분석했다.

**(1) 하남시 이용데이터 확인**  

![스크린샷 2025-01-24 223051.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/29c3299b-4ca4-4ffa-8e0e-9bf3bbf3abd2/c53de93c-e65a-45bb-b955-e69b08e79e9d/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7_2025-01-24_223051.png)

```
# 확인해보니 평균과 중앙값 차이가 크게 나지는 않지만, 중간중간 차이가 큰 부분이 많으므로 중앙값으로 계산
mean_values = count_h1.describe().loc["50%"]
```

```python
# 중앙값을 한 줄짜리 데이터프레임으로 변환
mean_df_h1 = pd.DataFrame([mean_values], columns=mean_values.index)
```

```python
mean_values = count_h2.describe().loc["mean"]

# 평균을 한 줄짜리 데이터프레임으로 변환
mean_df_h2 = pd.DataFrame([mean_values], columns=mean_values.index)
mean_df_h3 = count_h3 # 3대짜리는 데이터 하나임

mean_values = count_h5.describe().loc["mean"]

# 평균을 한 줄짜리 데이터프레임으로 변환
mean_df_h5 = pd.DataFrame([mean_values], columns=mean_values.index)
```

**(2) 차량 ‘1대’기준으로 ADF 테스트 진행**

시계열 데이터의 정상성 여부를 확인

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/29c3299b-4ca4-4ffa-8e0e-9bf3bbf3abd2/afb3d551-d7f8-49d1-b5ae-7e4c1c8cc224/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/29c3299b-4ca4-4ffa-8e0e-9bf3bbf3abd2/532ed200-177c-47af-89b8-589a623c0724/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/29c3299b-4ca4-4ffa-8e0e-9bf3bbf3abd2/acf18549-d04d-4d96-a849-164919b6cdc0/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/29c3299b-4ca4-4ffa-8e0e-9bf3bbf3abd2/3bfe329b-aada-49e6-9402-e4ab72923651/image.png)

- `p-value: 0.34053458821131316` 결과는 ADF 테스트에서 **귀무가설(H0)**을 기각하지 못한다.
- 차분 (Differencing) 혹은 계절성(Seasonality)을 제거으로 정상성 확보가 필요하다.

### **1. ACF 플롯 해석**

- **특징**: ACF가 1번째 시차 이후 급격히 감소하며, 이후 일부 시차에서 유의미한 자기상관성을 보인다.
- **결론**: 이는 MA(이동 평균) 모델의 **q값** 설정에 중요한 정보를 제공한다.
    - 1차 시차 이후 점진적으로 감소하는 패턴으로 보아, **q=1 또는 2**가 적절할 것으로 판단된다.

---

### **2. PACF 플롯 해석**

- **특징**: PACF는 1번째 시차에서 높은 유의미성을 보이고 이후 급격히 감소한다. 이는 AR(자기회귀) 모델의 p값 추정에 유용한 정보이다.
- **결론**: PACF의 절단점을 고려할 때 **p=1**이 적합할 것으로 보인다.

---

### **제안된 ARIMA 모델 파라미터**

- **ARIMA(p, d, q)** 설정:
    - **p (AR 차수)**: 1
    - **d (차분 차수)**: 이미 차분된 데이터를 사용했으므로 1차 차분(`d=1`)으로 설정
    - **q (MA 차수)**: 1 또는 2

이에 따라 다음 모델들을 우선적으로 검토했다:

- ARIMA(1, 1, 1)
- ARIMA(1, 1, 2)

**(3) ARIMA 모델 학습**

![스크린샷 2025-01-24 223844.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/29c3299b-4ca4-4ffa-8e0e-9bf3bbf3abd2/61b3c8d0-f658-4d2f-95ad-cd80fef60238/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7_2025-01-24_223844.png)

```python
forecast_steps = 365  # 365일 예측
forecast = fitted_model.forecast(steps=forecast_steps)
forecast_dates = pd.date_range(start=df_h1_t['시간'].iloc[-1], periods=forecast_steps + 1, freq='D')[1:]
```

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/29c3299b-4ca4-4ffa-8e0e-9bf3bbf3abd2/f78c0c91-5aa7-4428-a9f9-4780dfbbc2e5/image.png)

- 365일 예측결과 이용률이 상승되지 않는 결과 도출

**(4) 시계열 분해 후 결과 시각화**

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/29c3299b-4ca4-4ffa-8e0e-9bf3bbf3abd2/cb2a0e01-2ebb-4e94-afe5-7be7489373e3/image.png)

- 공공차량 이용 트렌드 경향성은 하락세를 보인다.
- 계절성의 주기는 50정도 확인되었다.

**(5) 계절성 효과를 반영한 SARIMAX 모델** 

     **학습 후 시각화**

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/29c3299b-4ca4-4ffa-8e0e-9bf3bbf3abd2/a8c7d54b-7f68-4783-aca6-145ab4239bdd/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/29c3299b-4ca4-4ffa-8e0e-9bf3bbf3abd2/c1000a71-e4fc-4849-a84e-cf3af81a7adf/image.png)

- 최대치를 넘는 예측 결과값이 없으며, 오히려 하락세를 띈다.
- 차량 ‘1대’의 세대수 지역의 규모를 늘릴 필요성이 없음을 확인했다.
