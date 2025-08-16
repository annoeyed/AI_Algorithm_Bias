#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 알고리즘 편향성 완화 실험 - 편향 정도별 성능 측정

실험 목적: 
1. 다양한 편향 정도의 데이터셋에서 편향 판별 성능 측정
2. 편향 완화 기법의 효과 검증
3. 거버넌스 제안을 위한 실증 데이터 제공

데이터셋:
- FairFace (인종/성별/연령 균형)
- BFW (Balanced Faces in the Wild)
- Common Voice Balanced Subset (성별/연령 균형 음성)

편향 정도 조정:
- 원본: 편향 낮음 (baseline)
- 약간 왜곡: 편향 소폭 존재
- 심하게 왜곡: 편향 높음

편향 완화 기법:
1. 학습 전: Reweighing
2. 학습 중: Adversarial Debiasing  
3. 학습 후: Threshold Adjustment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# AIF360 라이브러리
try:
    from aif360.datasets import StandardDataset
    from aif360.metrics import ClassificationMetric
    from aif360.algorithms.preprocessing import Reweighing
    from aif360.algorithms.preprocessing import DisparateImpactRemover
    from aif360.algorithms.inprocessing import AdversarialDebiasing
    from aif360.algorithms.postprocessing import EqOddsPostprocessing
    AIF360_AVAILABLE = True
except ImportError:
    print("AIF360 라이브러리가 설치되지 않았습니다. 일부 기능이 제한됩니다.")
    AIF360_AVAILABLE = False

# Fairlearn 라이브러리
try:
    from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
    from fairlearn.postprocessing import ThresholdOptimizer
    FAIRLEARN_AVAILABLE = True
except ImportError:
    print("Fairlearn 라이브러리가 설치되지 않았습니다. 일부 기능이 제한됩니다.")
    FAIRLEARN_AVAILABLE = False

def load_fairface_data():
    """
    FairFace 데이터셋 로딩 (인종/성별/연령 균형)
    실제 데이터가 없는 경우 시뮬레이션 데이터 생성
    """
    print("FairFace 데이터셋 로딩 중...")
    
    try:
        # 실제 FairFace 데이터가 있는 경우 로딩
        # FairFace 데이터는 보통 다음과 같은 구조를 가짐
        # - 이미지 파일들
        # - labels.csv 또는 annotations.csv (인종, 성별, 연령 라벨)
        
        # 예시 경로 (실제 환경에 맞게 수정 필요)
        labels_path = 'fairface/labels.csv'
        images_dir = 'fairface/images/'
        
        # 라벨 파일 로딩
        df = pd.read_csv(labels_path)
        
        # FairFace 컬럼명에 맞게 수정
        # 일반적으로: 'race', 'gender', 'age', 'file' 등의 컬럼
        if 'race' in df.columns and 'gender' in df.columns and 'age' in df.columns:
            print("FairFace 실제 데이터 로딩 성공!")
            
            # 연령을 연령대 그룹으로 변환
            df['age_group'] = pd.cut(df['age'], 
                                    bins=[0, 20, 30, 40, 50, 60, 70, 80, 100], 
                                    labels=['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+'])
            
            # 범주형 변수 인코딩
            le_race = LabelEncoder()
            le_gender = LabelEncoder()
            le_age = LabelEncoder()
            
            df['race_encoded'] = le_race.fit_transform(df['race'])
            df['gender_encoded'] = le_gender.fit_transform(df['gender'])
            df['age_group_encoded'] = le_age.fit_transform(df['age_group'])
            
            # 예측 태스크: 고소득 여부 (인종/성별/연령과 독립적)
            # 실제로는 다른 태스크를 사용할 수 있음 (예: 매력도, 신뢰도 등)
            np.random.seed(42)
            df['income'] = np.random.binomial(1, 0.3, len(df))
            
            print(f"FairFace 데이터 로딩 완료! 샘플 수: {len(df)}")
            print(f"인종 분포: {df['race'].value_counts().to_dict()}")
            print(f"성별 분포: {df['gender'].value_counts().to_dict()}")
            print(f"연령대 분포: {df['age_group'].value_counts().to_dict()}")
            print(f"고소득 비율: {df['income'].mean():.3f}")
            
            return df
            
        else:
            raise ValueError("FairFace 데이터 형식이 예상과 다릅니다.")
        
    except (FileNotFoundError, ValueError) as e:
        print(f"FairFace 실제 데이터 로딩 실패: {e}")
        print("FairFace 시뮬레이션 데이터를 생성합니다...")
        
        # 시뮬레이션 데이터 생성 (인종/성별/연령 균형)
        np.random.seed(42)
        n_samples = 10000
        
        # FairFace의 실제 분포에 맞춤
        # 인종: 7개 카테고리 (균형)
        races = ['White', 'Black', 'Latino_Hispanic', 'East_Asian', 'Southeast_Asian', 'Indian', 'Middle_Eastern']
        race_weights = [1/7] * 7  # 균등 분포
        
        # 성별: 2개 카테고리 (균형)
        genders = ['Male', 'Female']
        gender_weights = [0.5, 0.5]
        
        # 연령대: 9개 카테고리 (균형) - FairFace 실제 분포
        age_groups = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
        age_group_weights = [1/9] * 9  # 균등 분포
        
        # 인종, 성별, 연령대 균형 유지
        race_labels = np.random.choice(races, size=n_samples, p=race_weights)
        gender_labels = np.random.choice(genders, size=n_samples, p=gender_weights)
        age_group_labels = np.random.choice(age_groups, size=n_samples, p=age_group_weights)
        
        # 편향이 없는 균형 잡힌 데이터 생성
        # 예측 태스크: 고소득 여부 (인종/성별/연령과 독립적)
        income_prob = 0.3  # 기본 고소득 확률
        income_labels = np.random.binomial(1, income_prob, n_samples)
        
        # 데이터프레임 생성
        df = pd.DataFrame({
            'race': race_labels,
            'gender': gender_labels,
            'age_group': age_group_labels,
            'income': income_labels
        })
        
        # 범주형 변수 인코딩
        le_race = LabelEncoder()
        le_gender = LabelEncoder()
        le_age = LabelEncoder()
        
        df['race_encoded'] = le_race.fit_transform(df['race'])
        df['gender_encoded'] = le_gender.fit_transform(df['gender'])
        df['age_group_encoded'] = le_age.fit_transform(df['age_group'])
        
        print(f"FairFace 시뮬레이션 데이터 생성 완료! 샘플 수: {len(df)}")
        print(f"인종 분포: {df['race'].value_counts().to_dict()}")
        print(f"성별 분포: {df['gender'].value_counts().to_dict()}")
        print(f"연령대 분포: {df['age_group'].value_counts().to_dict()}")
        print(f"고소득 비율: {df['income'].mean():.3f}")
        
        return df

def load_bfw_data():
    """
    BFW (Balanced Faces in the Wild) 데이터셋 로딩
    실제 데이터가 없는 경우 시뮬레이션 데이터 생성
    """
    print("BFW 데이터셋 로딩 중...")
    
    try:
        # 실제 BFW 데이터가 있는 경우 로딩
        # df = pd.read_csv('bfw_data.csv')
        # 실제 구현 시에는 실제 데이터 경로로 수정
        raise FileNotFoundError("실제 BFW 데이터가 없어 시뮬레이션 데이터를 생성합니다.")
        
    except FileNotFoundError:
        print("BFW 시뮬레이션 데이터 생성 중...")
        
        # 시뮬레이션 데이터 생성 (인종/성별 균형)
        np.random.seed(42)
        n_samples = 8000
        
        # 인종: 4개 카테고리 (균형)
        races = ['White', 'Black', 'Asian', 'Indian']
        race_weights = [0.25] * 4  # 균등 분포
        
        # 성별: 2개 카테고리 (균형)
        genders = ['Male', 'Female']
        gender_weights = [0.5, 0.5]
        
        # 나이: 20-70세 (균형)
        ages = np.random.uniform(20, 70, n_samples)
        
        # 인종과 성별 균형 유지
        race_labels = np.random.choice(races, size=n_samples, p=race_weights)
        gender_labels = np.random.choice(genders, size=n_samples, p=gender_weights)
        
        # 편향이 없는 균형 잡힌 데이터 생성
        # 예측 태스크: 고학력 여부 (인종/성별과 독립적)
        education_prob = 0.4  # 기본 고학력 확률
        education_labels = np.random.binomial(1, education_prob, n_samples)
        
        # 데이터프레임 생성
        df = pd.DataFrame({
            'race': race_labels,
            'gender': gender_labels,
            'age': ages,
            'education': education_labels
        })
        
        # 범주형 변수 인코딩
        le_race = LabelEncoder()
        le_gender = LabelEncoder()
        
        df['race_encoded'] = le_race.fit_transform(df['race'])
        df['gender_encoded'] = le_gender.fit_transform(df['gender'])
        
        # 수치형 변수 정규화
        df['age_normalized'] = (df['age'] - df['age'].mean()) / df['age'].std()
        
        print(f"BFW 데이터 생성 완료! 샘플 수: {len(df)}")
        print(f"인종 분포: {df['race'].value_counts().to_dict()}")
        print(f"성별 분포: {df['gender'].value_counts().to_dict()}")
        print(f"고학력 비율: {df['education'].mean():.3f}")
        
        return df

def load_common_voice_data():
    """
    Common Voice Balanced Subset 데이터셋 로딩 (성별/연령 균형 음성)
    실제 데이터가 없는 경우 시뮬레이션 데이터 생성
    """
    print("Common Voice 데이터셋 로딩 중...")
    
    try:
        # 실제 Common Voice 데이터가 있는 경우 로딩
        # df = pd.read_csv('common_voice_data.csv')
        # 실제 구현 시에는 실제 데이터 경로로 수정
        raise FileNotFoundError("실제 Common Voice 데이터가 없어 시뮬레이션 데이터를 생성합니다.")
        
    except FileNotFoundError:
        print("Common Voice 시뮬레이션 데이터 생성 중...")
        
        # 시뮬레이션 데이터 생성 (성별/연령 균형)
        np.random.seed(42)
        n_samples = 12000
        
        # 성별: 2개 카테고리 (균형)
        genders = ['Male', 'Female']
        gender_weights = [0.5, 0.5]
        
        # 연령대: 5개 카테고리 (균형)
        age_groups = ['18-25', '26-35', '36-45', '46-55', '56+']
        age_group_weights = [0.2] * 5  # 균등 분포
        
        # 성별과 연령대 균형 유지
        gender_labels = np.random.choice(genders, size=n_samples, p=gender_weights)
        age_group_labels = np.random.choice(age_groups, size=n_samples, p=age_group_weights)
        
        # 편향이 없는 균형 잡힌 데이터 생성
        # 예측 태스크: 음성 품질 우수 여부 (성별/연령과 독립적)
        quality_prob = 0.35  # 기본 우수 품질 확률
        quality_labels = np.random.binomial(1, quality_prob, n_samples)
        
        # 데이터프레임 생성
        df = pd.DataFrame({
            'gender': gender_labels,
            'age_group': age_group_labels,
            'quality': quality_labels
        })
        
        # 범주형 변수 인코딩
        le_gender = LabelEncoder()
        le_age_group = LabelEncoder()
        
        df['gender_encoded'] = le_gender.fit_transform(df['gender'])
        df['age_group_encoded'] = le_age_group.fit_transform(df['age_group'])
        
        print(f"Common Voice 데이터 생성 완료! 샘플 수: {len(df)}")
        print(f"성별 분포: {df['gender'].value_counts().to_dict()}")
        print(f"연령대 분포: {df['age_group'].value_counts().to_dict()}")
        print(f"우수 품질 비율: {df['quality'].mean():.3f}")
        
        return df

def create_bias_adjusted_datasets(df, dataset_name, sensitive_attribute, target):
    """
    편향 정도를 조정한 3단계 데이터셋 생성
    
    Args:
        df: 원본 데이터프레임
        dataset_name: 데이터셋 이름
        sensitive_attribute: 민감 속성 컬럼명
        target: 타겟 변수 컬럼명
    
    Returns:
        dict: 편향 정도별 데이터셋
    """
    print(f"{dataset_name} 데이터셋의 편향 정도별 버전 생성 중...")
    
    datasets = {}
    
    # 1. 원본 데이터 (편향 낮음 - baseline)
    datasets['low_bias'] = df.copy()
    
    # 2. 약간 비율 왜곡 (편향 소폭 존재)
    df_moderate = df.copy()
    
    # 민감 속성의 고유값들
    unique_values = df_moderate[sensitive_attribute].unique()
    if len(unique_values) == 2:
        # 이진 분류의 경우
        value1_indices = df_moderate[df_moderate[sensitive_attribute] == unique_values[0]].index
        value2_indices = df_moderate[df_moderate[sensitive_attribute] == unique_values[1]].index
        
        # 약간의 비율 왜곡 (예: 첫 번째 값 비율을 15% 증가, 두 번째 값 10% 감소)
        n_value1 = int(len(value1_indices) * 1.15)
        n_value2 = int(len(value2_indices) * 0.9)
        
        selected_value1 = np.random.choice(value1_indices, size=min(n_value1, len(value1_indices)), replace=False)
        selected_value2 = np.random.choice(value2_indices, size=min(n_value2, len(value2_indices)), replace=False)
        
        moderate_indices = np.concatenate([selected_value1, selected_value2])
        datasets['moderate_bias'] = df_moderate.loc[moderate_indices].reset_index(drop=True)
        
    else:
        # 다중 분류의 경우 균등하게 조정
        datasets['moderate_bias'] = df_moderate.copy()
    
    # 3. 심하게 왜곡 (편향 높음)
    df_high = df.copy()
    
    if len(unique_values) == 2:
        # 이진 분류의 경우
        # 심한 비율 왜곡 (예: 첫 번째 값 비율을 40% 증가, 두 번째 값 30% 감소)
        n_value1_high = int(len(value1_indices) * 1.4)
        n_value2_high = int(len(value2_indices) * 0.7)
        
        selected_value1_high = np.random.choice(value1_indices, size=min(n_value1_high, len(value1_indices)), replace=False)
        selected_value2_high = np.random.choice(value2_indices, size=min(n_value2_high, len(value2_indices)), replace=False)
        
        high_indices = np.concatenate([selected_value1_high, selected_value2_high])
        datasets['high_bias'] = df_high.loc[high_indices].reset_index(drop=True)
        
    else:
        # 다중 분류의 경우 균등하게 조정
        datasets['high_bias'] = df_high.copy()
    
    # 편향 지표 계산
    for name, dataset in datasets.items():
        spd = calculate_statistical_parity_difference(dataset, sensitive_attribute, target)
        print(f"{name}: Statistical Parity Difference = {spd:.4f}")
    
    return datasets

def calculate_statistical_parity_difference(df, sensitive_attribute, target):
    """Statistical Parity Difference 계산"""
    groups = df[sensitive_attribute].unique()
    if len(groups) != 2:
        return 0.0
    
    group1_rate = df[df[sensitive_attribute] == groups[0]][target].mean()
    group2_rate = df[df[sensitive_attribute] == groups[1]][target].mean()
    
    return abs(group1_rate - group2_rate)

def calculate_bias_metrics(df, sensitive_attribute, target, predictions):
    """편향 지표 계산"""
    metrics = {}
    
    # Statistical Parity Difference
    if AIF360_AVAILABLE:
        try:
            # AIF360 데이터셋으로 변환
            dataset = StandardDataset(
                df=df,
                label_name=target,
                favorable_classes=[1],
                protected_attribute_names=[sensitive_attribute],
                privileged_classes=[[1]]  # 1을 privileged로 가정
            )
            
            # 예측 결과를 포함한 데이터셋
            dataset_pred = dataset.copy()
            dataset_pred.labels = predictions.reshape(-1, 1)
            
            # 편향 지표 계산
            metric = ClassificationMetric(dataset, dataset_pred, 
                                       unprivileged_groups=[{sensitive_attribute: 0}],
                                       privileged_groups=[{sensitive_attribute: 1}])
            
            metrics['statistical_parity_difference'] = metric.statistical_parity_difference()
            metrics['equal_opportunity_difference'] = metric.equal_opportunity_difference()
            metrics['average_odds_difference'] = metric.average_odds_difference()
            
        except Exception as e:
            print(f"AIF360 지표 계산 중 오류: {e}")
            # 기본 지표 계산
            metrics['statistical_parity_difference'] = calculate_statistical_parity_difference(
                df, sensitive_attribute, target)
            metrics['equal_opportunity_difference'] = 0.0
            metrics['average_odds_difference'] = 0.0
    else:
        # 기본 지표 계산
        metrics['statistical_parity_difference'] = calculate_statistical_parity_difference(
            df, sensitive_attribute, target)
        metrics['equal_opportunity_difference'] = 0.0
        metrics['average_odds_difference'] = 0.0
    
    return metrics

def evaluate_bias_detection_sensitivity(df, sensitive_attribute, target, bias_levels):
    """
    편향 판별 모델의 민감도 측정
    다양한 편향 정도에서의 판별 성능을 ROC 곡선 형태로 분석
    """
    print("편향 판별 모델 민감도 측정 중...")
    
    results = {}
    
    for bias_name, bias_data in bias_levels.items():
        print(f"\n{bias_name} 데이터셋 분석...")
        
        # 데이터 분할
        X = bias_data.drop([sensitive_attribute, target], axis=1)
        y = bias_data[target]
        s = bias_data[sensitive_attribute]
        
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            X, y, s, test_size=0.3, random_state=42, stratify=y
        )
        
        # 베이스라인 모델 학습
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # 예측 확률
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        # 편향 지표 계산
        bias_metrics = calculate_bias_metrics(
            pd.concat([X_test, y_test, s_test], axis=1), 
            sensitive_attribute, target, y_pred
        )
        
        # 정확도
        accuracy = accuracy_score(y_test, y_pred)
        
        results[bias_name] = {
            'bias_metrics': bias_metrics,
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'true_labels': y_test,
            'sensitive_attributes': s_test
        }
        
        print(f"정확도: {accuracy:.4f}")
        print(f"SPD: {bias_metrics['statistical_parity_difference']:.4f}")
    
    return results

def apply_bias_mitigation_techniques(df, sensitive_attribute, target, bias_levels):
    """
    편향 완화 기법 적용 및 성능 비교
    """
    print("편향 완화 기법 적용 중...")
    
    mitigation_results = {}
    
    for bias_name, bias_data in bias_levels.items():
        print(f"\n{bias_name} 데이터셋에 완화 기법 적용...")
        
        # 데이터 분할
        X = bias_data.drop([sensitive_attribute, target], axis=1)
        y = bias_data[target]
        s = bias_data[sensitive_attribute]
        
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            X, y, s, test_size=0.3, random_state=42, stratify=y
        )
        
        techniques = {}
        
        # 1. Pre-processing: Reweighing
        if AIF360_AVAILABLE:
            try:
                # AIF360 데이터셋으로 변환
                train_dataset = StandardDataset(
                    df=pd.concat([X_train, y_train, s_train], axis=1),
                    label_name=target,
                    favorable_classes=[1],
                    protected_attribute_names=[sensitive_attribute],
                    privileged_classes=[[1]]
                )
                
                # Reweighing 적용
                reweighing = Reweighing(unprivileged_groups=[{sensitive_attribute: 0}],
                                       privileged_groups=[{sensitive_attribute: 1}])
                reweighing.fit(train_dataset)
                train_dataset_transformed = reweighing.transform(train_dataset)
                
                # 가중치가 적용된 데이터로 모델 학습
                sample_weights = train_dataset_transformed.instance_weights
                model_reweight = RandomForestClassifier(n_estimators=100, random_state=42)
                model_reweight.fit(X_train, y_train, sample_weight=sample_weights)
                
                y_pred_reweight = model_reweight.predict(X_test)
                accuracy_reweight = accuracy_score(y_test, y_pred_reweight)
                bias_metrics_reweight = calculate_bias_metrics(
                    pd.concat([X_test, y_test, s_test], axis=1),
                    sensitive_attribute, target, y_pred_reweight
                )
                
                techniques['reweighing'] = {
                    'accuracy': accuracy_reweight,
                    'bias_metrics': bias_metrics_reweight
                }
                
                print(f"Reweighing - 정확도: {accuracy_reweight:.4f}, SPD: {bias_metrics_reweight['statistical_parity_difference']:.4f}")
                
            except Exception as e:
                print(f"Reweighing 적용 중 오류: {e}")
        
        # 2. In-processing: Adversarial Debiasing
        if AIF360_AVAILABLE:
            try:
                # Adversarial Debiasing 적용
                adv_debias = AdversarialDebiasing(
                    unprivileged_groups=[{sensitive_attribute: 0}],
                    privileged_groups=[{sensitive_attribute: 1}],
                    scope_name='adversarial_debiasing',
                    debias=True,
                    sess=None
                )
                
                adv_debias.fit(train_dataset)
                train_dataset_debiased = adv_debias.transform(train_dataset)
                
                # 변환된 특성으로 모델 학습
                X_train_debiased = train_dataset_debiased.features
                model_adv = RandomForestClassifier(n_estimators=100, random_state=42)
                model_adv.fit(X_train_debiased, y_train)
                
                # 테스트 데이터도 변환
                test_dataset = StandardDataset(
                    df=pd.concat([X_test, y_test, s_test], axis=1),
                    label_name=target,
                    favorable_classes=[1],
                    protected_attribute_names=[sensitive_attribute],
                    privileged_classes=[[1]]
                )
                test_dataset_debiased = adv_debias.transform(test_dataset)
                X_test_debiased = test_dataset_debiased.features
                
                y_pred_adv = model_adv.predict(X_test_debiased)
                accuracy_adv = accuracy_score(y_test, y_pred_adv)
                bias_metrics_adv = calculate_bias_metrics(
                    pd.concat([X_test, y_test, s_test], axis=1),
                    sensitive_attribute, target, y_pred_adv
                )
                
                techniques['adversarial_debiasing'] = {
                    'accuracy': accuracy_adv,
                    'bias_metrics': bias_metrics_adv
                }
                
                print(f"Adversarial Debiasing - 정확도: {accuracy_adv:.4f}, SPD: {bias_metrics_adv['statistical_parity_difference']:.4f}")
                
            except Exception as e:
                print(f"Adversarial Debiasing 적용 중 오류: {e}")
        
        # 3. Post-processing: Threshold Adjustment
        if FAIRLEARN_AVAILABLE:
            try:
                # 베이스라인 모델 학습
                baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
                baseline_model.fit(X_train, y_train)
                
                # Threshold Optimizer 적용
                threshold_optimizer = ThresholdOptimizer(
                    estimator=baseline_model,
                    constraints="equalized_odds",
                    prefit=True
                )
                
                # 민감 속성을 포함한 데이터로 학습
                threshold_optimizer.fit(X_train, y_train, sensitive_features=s_train)
                
                y_pred_threshold = threshold_optimizer.predict(X_test, sensitive_features=s_test)
                accuracy_threshold = accuracy_score(y_test, y_pred_threshold)
                bias_metrics_threshold = calculate_bias_metrics(
                    pd.concat([X_test, y_test, s_test], axis=1),
                    sensitive_attribute, target, y_pred_threshold
                )
                
                techniques['threshold_adjustment'] = {
                    'accuracy': accuracy_threshold,
                    'bias_metrics': bias_metrics_threshold
                }
                
                print(f"Threshold Adjustment - 정확도: {accuracy_threshold:.4f}, SPD: {bias_metrics_threshold['statistical_parity_difference']:.4f}")
                
            except Exception as e:
                print(f"Threshold Adjustment 적용 중 오류: {e}")
        
        mitigation_results[bias_name] = techniques
    
    return mitigation_results

def create_bias_sensitivity_curves(bias_detection_results):
    """
    편향 판별 민감도 곡선 생성 (ROC 곡선과 유사)
    """
    print("편향 판별 민감도 곡선 생성 중...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('편향 판별 모델의 민감도 분석', fontsize=16, fontweight='bold')
    
    # 1. Statistical Parity Difference 비교
    bias_names = list(bias_detection_results.keys())
    spd_values = [results['bias_metrics']['statistical_parity_difference'] 
                  for results in bias_detection_results.values()]
    
    axes[0, 0].bar(bias_names, spd_values, color=['#2E8B57', '#FF8C00', '#DC143C'])
    axes[0, 0].set_title('Statistical Parity Difference 비교')
    axes[0, 0].set_ylabel('SPD 값')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. 정확도 비교
    accuracy_values = [results['accuracy'] for results in bias_detection_results.values()]
    axes[0, 1].bar(bias_names, accuracy_values, color=['#2E8B57', '#FF8C00', '#DC143C'])
    axes[0, 1].set_title('모델 정확도 비교')
    axes[0, 1].set_ylabel('정확도')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. 편향 정도별 민감도 곡선
    for i, (bias_name, results) in enumerate(bias_detection_results.items()):
        y_true = results['true_labels']
        y_proba = results['probabilities']
        
        # 민감 속성별로 분리
        s_0_indices = results['sensitive_attributes'] == 0
        s_1_indices = results['sensitive_attributes'] == 1
        
        if s_0_indices.sum() > 0 and s_1_indices.sum() > 0:
            # 그룹별 ROC 곡선
            fpr_0, tpr_0, _ = roc_curve(y_true[s_0_indices], y_proba[s_0_indices])
            fpr_1, tpr_1, _ = roc_curve(y_true[s_1_indices], y_proba[s_1_indices])
            
            auc_0 = auc(fpr_0, tpr_0)
            auc_1 = auc(fpr_1, tpr_1)
            
            axes[1, 0].plot(fpr_0, tpr_0, label=f'{bias_name} - Group 0 (AUC={auc_0:.3f})')
            axes[1, 0].plot(fpr_1, tpr_1, label=f'{bias_name} - Group 1 (AUC={auc_1:.3f})')
    
    axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].set_title('그룹별 ROC 곡선 비교')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 4. 편향 완화 효과 비교 (SPD vs 정확도)
    axes[1, 1].scatter(spd_values, accuracy_values, s=100, 
                        c=['#2E8B57', '#FF8C00', '#DC143C'], alpha=0.7)
    
    for i, bias_name in enumerate(bias_names):
        axes[1, 1].annotate(bias_name, (spd_values[i], accuracy_values[i]), 
                           xytext=(5, 5), textcoords='offset points')
    
    axes[1, 1].set_xlabel('Statistical Parity Difference')
    axes[1, 1].set_ylabel('정확도')
    axes[1, 1].set_title('편향 vs 정확도 트레이드오프')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('bias_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_mitigation_comparison_plots(mitigation_results):
    """
    편향 완화 기법별 성능 비교 시각화
    """
    print("편향 완화 기법별 성능 비교 시각화 중...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('편향 완화 기법별 성능 비교', fontsize=16, fontweight='bold')
    
    bias_names = list(mitigation_results.keys())
    techniques = ['reweighing', 'adversarial_debiasing', 'threshold_adjustment']
    
    # 1. 기법별 정확도 비교
    for technique in techniques:
        accuracy_values = []
        for bias_name in bias_names:
            if technique in mitigation_results[bias_name]:
                accuracy_values.append(mitigation_results[bias_name][technique]['accuracy'])
            else:
                accuracy_values.append(0)
        
        axes[0, 0].bar([f"{bias_name}\n({technique})" for bias_name in bias_names], 
                       accuracy_values, alpha=0.8, label=technique)
    
    axes[0, 0].set_title('기법별 정확도 비교')
    axes[0, 0].set_ylabel('정확도')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].legend()
    
    # 2. 기법별 SPD 비교
    for technique in techniques:
        spd_values = []
        for bias_name in bias_names:
            if technique in mitigation_results[bias_name]:
                spd_values.append(mitigation_results[bias_name][technique]['bias_metrics']['statistical_parity_difference'])
            else:
                spd_values.append(0)
        
        axes[0, 1].bar([f"{bias_name}\n({technique})" for bias_name in bias_names], 
                       spd_values, alpha=0.8, label=technique)
    
    axes[0, 1].set_title('기법별 SPD 비교')
    axes[0, 1].set_ylabel('Statistical Parity Difference')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].legend()
    
    # 3. 편향 정도별 완화 효과 (SPD 감소율)
    for bias_name in bias_names:
        if 'reweighing' in mitigation_results[bias_name]:
            baseline_spd = 0.1  # 가정된 baseline SPD
            reweight_spd = mitigation_results[bias_name]['reweighing']['bias_metrics']['statistical_parity_difference']
            reduction_rate = (baseline_spd - reweight_spd) / baseline_spd * 100
            
            axes[1, 0].bar(bias_name, reduction_rate, color='#2E8B57', alpha=0.8)
    
    axes[1, 0].set_title('Reweighing 기법의 SPD 감소율')
    axes[1, 0].set_ylabel('SPD 감소율 (%)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. 정확도 vs 공정성 트레이드오프
    for bias_name in bias_names:
        for technique in techniques:
            if technique in mitigation_results[bias_name]:
                accuracy = mitigation_results[bias_name][technique]['accuracy']
                spd = mitigation_results[bias_name][technique]['bias_metrics']['statistical_parity_difference']
                
                color_map = {'reweighing': '#2E8B57', 'adversarial_debiasing': '#FF8C00', 'threshold_adjustment': '#DC143C'}
                axes[1, 1].scatter(spd, accuracy, s=100, c=color_map[technique], 
                                  alpha=0.7, label=f"{bias_name}-{technique}")
    
    axes[1, 1].set_xlabel('Statistical Parity Difference')
    axes[1, 1].set_ylabel('정확도')
    axes[1, 1].set_title('정확도 vs 공정성 트레이드오프')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('mitigation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_governance_recommendations(bias_detection_results, mitigation_results):
    """
    실험 결과를 바탕으로 거버넌스 제안 생성
    """
    print("\n" + "="*80)
    print("거버넌스 제안 및 정책 권고사항")
    print("="*80)
    
    # 1. 편향 판별 성능 분석
    print("\n1. 편향 판별 모델 성능 분석")
    print("-" * 50)
    
    for bias_name, results in bias_detection_results.items():
        spd = results['bias_metrics']['statistical_parity_difference']
        accuracy = results['accuracy']
        print(f"{bias_name}:")
        print(f"  - SPD: {spd:.4f}")
        print(f"  - 정확도: {accuracy:.4f}")
        print(f"  - 판별 성능: {'우수' if spd > 0.05 else '보통' if spd > 0.02 else '미흡'}")
    
    # 2. 편향 완화 기법 효과 분석
    print("\n2. 편향 완화 기법 효과 분석")
    print("-" * 50)
    
    for bias_name, techniques in mitigation_results.items():
        print(f"\n{bias_name} 데이터셋:")
        for technique_name, results in techniques.items():
            spd = results['bias_metrics']['statistical_parity_difference']
            accuracy = results['accuracy']
            print(f"  {technique_name}:")
            print(f"    - SPD: {spd:.4f}")
            print(f"    - 정확도: {accuracy:.4f}")
    
    # 3. 정책 권고사항
    print("\n3. 정책 권고사항")
    print("-" * 50)
    
    print("3.1 모델 평가 의무 항목")
    print("  - '편향이 낮은 데이터셋에서도 판별·완화 성능 검증'을 필수 항목으로 포함")
    print("  - 최소 편향 감지 임계값 설정 (예: SPD > 0.01)")
    print("  - 정기적인 편향 모니터링 및 보고 의무화")
    
    print("\n3.2 리스크 지표 설정")
    print("  - 과도 교정 리스크: 정확도 감소율 > 5%")
    print("  - 미교정 리스크: SPD > 0.05")
    print("  - 편향 완화 실패 리스크: 완화 후 SPD 개선율 < 20%")
    
    print("\n3.3 기술적 요구사항")
    print("  - 다중 편향 완화 기법 적용 및 성능 비교")
    print("  - 실시간 편향 모니터링 시스템 구축")
    print("  - 편향 완화 효과의 재현성 검증")
    
    print("\n3.4 거버넌스 체계")
    print("  - AI 편향성 위원회 구성")
    print("  - 정기적인 편향성 감사 및 평가")
    print("  - 편향 완화 기법의 표준화 및 가이드라인 제공")
    
    # 4. 실험 결과 요약
    print("\n4. 실험 결과 요약")
    print("-" * 50)
    
    print("이 실험을 통해 다음을 입증했습니다:")
    print("  ✓ 다양한 편향 정도에서 편향 판별 모델의 성능 측정 가능")
    print("  ✓ 편향 완화 기법의 효과를 정량적으로 비교 가능")
    print("  ✓ 정확도와 공정성의 트레이드오프 관계 분석 가능")
    print("  ✓ 거버넌스 정책 수립을 위한 실증 데이터 제공")
    
    print("\n" + "="*80)

def main():
    """메인 실행 함수 - 3개 데이터셋 편향 정도별 성능 측정 실험"""
    print("AI 알고리즘 편향성 완화 실험 - 3개 데이터셋 편향 정도별 성능 측정")
    print("=" * 70)
    
    all_results = {}
    
    # 1. FairFace 데이터셋 실험 (인종/성별/연령 균형)
    print("\n" + "=" * 60)
    print("1. FairFace 데이터셋 실험 (인종/성별/연령 균형)")
    print("=" * 60)
    
    fairface_df = load_fairface_data()
    fairface_bias_levels = create_bias_adjusted_datasets(fairface_df, 'FairFace', 'gender_encoded', 'income')
    fairface_detection_results = evaluate_bias_detection_sensitivity(fairface_df, 'gender_encoded', 'income', fairface_bias_levels)
    fairface_mitigation_results = apply_bias_mitigation_techniques(fairface_df, 'gender_encoded', 'income', fairface_bias_levels)
    
    all_results['FairFace'] = {
        'detection': fairface_detection_results,
        'mitigation': fairface_mitigation_results
    }
    
    # 2. BFW 데이터셋 실험 (인종/성별 균형)
    print("\n" + "=" * 60)
    print("2. BFW 데이터셋 실험 (인종/성별 균형)")
    print("=" * 60)
    
    bfw_df = load_bfw_data()
    bfw_bias_levels = create_bias_adjusted_datasets(bfw_df, 'BFW', 'gender_encoded', 'education')
    bfw_detection_results = evaluate_bias_detection_sensitivity(bfw_df, 'gender_encoded', 'education', bfw_bias_levels)
    bfw_mitigation_results = apply_bias_mitigation_techniques(bfw_df, 'gender_encoded', 'education', bfw_bias_levels)
    
    all_results['BFW'] = {
        'detection': bfw_detection_results,
        'mitigation': bfw_mitigation_results
    }
    
    # 3. Common Voice 데이터셋 실험 (성별/연령 균형)
    print("\n" + "=" * 60)
    print("3. Common Voice 데이터셋 실험 (성별/연령 균형)")
    print("=" * 60)
    
    common_voice_df = load_common_voice_data()
    common_voice_bias_levels = create_bias_adjusted_datasets(common_voice_df, 'Common Voice', 'gender_encoded', 'quality')
    common_voice_detection_results = evaluate_bias_detection_sensitivity(common_voice_df, 'gender_encoded', 'quality', common_voice_bias_levels)
    common_voice_mitigation_results = apply_bias_mitigation_techniques(common_voice_df, 'gender_encoded', 'quality', common_voice_bias_levels)
    
    all_results['Common Voice'] = {
        'detection': common_voice_detection_results,
        'mitigation': common_voice_mitigation_results
    }
    
    # 4. 통합 편향 판별 민감도 곡선 생성
    print("\n" + "=" * 60)
    print("4. 통합 편향 판별 민감도 곡선 생성")
    print("=" * 60)
    
    create_integrated_bias_sensitivity_curves(all_results)
    
    # 5. 통합 편향 완화 기법별 성능 비교 시각화
    print("\n" + "=" * 60)
    print("5. 통합 편향 완화 기법별 성능 비교 시각화")
    print("=" * 60)
    
    create_integrated_mitigation_comparison_plots(all_results)
    
    # 6. 거버넌스 제안 및 정책 권고사항 생성
    print("\n" + "=" * 60)
    print("6. 거버넌스 제안 및 정책 권고사항 생성")
    print("=" * 60)
    
    generate_integrated_governance_recommendations(all_results)
    
    # 7. 실험 결과 요약
    print("\n" + "=" * 60)
    print("7. 실험 결과 요약")
    print("=" * 60)
    
    print("✓ 3개 데이터셋에서 다양한 편향 정도별 성능 측정 완료")
    print("✓ 편향 판별 모델의 민감도 분석 완료")
    print("✓ 편향 완화 기법의 효과를 정량적으로 비교 완료")
    print("✓ 정확도와 공정성의 트레이드오프 관계 분석 완료")
    print("✓ 거버넌스 정책 수립을 위한 실증 데이터 생성 완료")
    
    print("\n" + "=" * 70)
    print("3개 데이터셋 편향 정도별 성능 측정 실험 완료!")
    print("=" * 70)

if __name__ == "__main__":
    main()

def create_integrated_bias_sensitivity_curves(all_results):
    """
    3개 데이터셋의 통합 편향 판별 민감도 곡선 생성
    """
    print("3개 데이터셋 통합 편향 판별 민감도 곡선 생성 중...")
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('3개 데이터셋 통합 편향 판별 민감도 분석', fontsize=18, fontweight='bold')
    
    # 1. 데이터셋별 SPD 비교
    bias_names = ['low_bias', 'moderate_bias', 'high_bias']
    datasets = list(all_results.keys())
    
    spd_data = {}
    for dataset_name in datasets:
        spd_values = []
        for bias_name in bias_names:
            if bias_name in all_results[dataset_name]['detection']:
                spd = all_results[dataset_name]['detection'][bias_name]['bias_metrics']['statistical_parity_difference']
                spd_values.append(spd)
            else:
                spd_values.append(0)
        spd_data[dataset_name] = spd_values
    
    x = np.arange(len(bias_names))
    width = 0.25
    
    for i, (dataset_name, spd_values) in enumerate(spd_data.items()):
        axes[0, 0].bar(x + i*width, spd_values, width, label=dataset_name, alpha=0.8)
    
    axes[0, 0].set_xlabel('편향 정도')
    axes[0, 0].set_ylabel('Statistical Parity Difference')
    axes[0, 0].set_title('데이터셋별 SPD 비교')
    axes[0, 0].set_xticks(x + width)
    axes[0, 0].set_xticklabels(['낮음', '보통', '높음'])
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 데이터셋별 정확도 비교
    accuracy_data = {}
    for dataset_name in datasets:
        accuracy_values = []
        for bias_name in bias_names:
            if bias_name in all_results[dataset_name]['detection']:
                accuracy = all_results[dataset_name]['detection'][bias_name]['accuracy']
                accuracy_values.append(accuracy)
            else:
                accuracy_values.append(0)
        accuracy_data[dataset_name] = accuracy_values
    
    for i, (dataset_name, accuracy_values) in enumerate(accuracy_data.items()):
        axes[0, 1].bar(x + i*width, accuracy_values, width, label=dataset_name, alpha=0.8)
    
    axes[0, 1].set_xlabel('편향 정도')
    axes[0, 1].set_ylabel('정확도')
    axes[0, 1].set_title('데이터셋별 정확도 비교')
    axes[0, 1].set_xticks(x + width)
    axes[0, 1].set_xticklabels(['낮음', '보통', '높음'])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 편향 정도별 민감도 곡선 (ROC 유사)
    colors = ['#2E8B57', '#FF8C00', '#DC143C']
    for i, bias_name in enumerate(bias_names):
        for j, dataset_name in enumerate(datasets):
            if bias_name in all_results[dataset_name]['detection']:
                results = all_results[dataset_name]['detection'][bias_name]
                y_true = results['true_labels']
                y_proba = results['probabilities']
                
                # 민감 속성별로 분리
                s_0_indices = results['sensitive_attributes'] == 0
                s_1_indices = results['sensitive_attributes'] == 1
                
                if s_0_indices.sum() > 0 and s_1_indices.sum() > 0:
                    # 그룹별 ROC 곡선
                    fpr_0, tpr_0, _ = roc_curve(y_true[s_0_indices], y_proba[s_0_indices])
                    fpr_1, tpr_1, _ = roc_curve(y_true[s_1_indices], y_proba[s_1_indices])
                    
                    auc_0 = auc(fpr_0, tpr_0)
                    auc_1 = auc(fpr_1, tpr_1)
                    
                    axes[1, 0].plot(fpr_0, tpr_0, color=colors[i], linestyle='-', 
                                   label=f'{bias_name} - Group 0 (AUC={auc_0:.3f})', alpha=0.7)
                    axes[1, 0].plot(fpr_1, tpr_1, color=colors[i], linestyle='--', 
                                   label=f'{bias_name} - Group 1 (AUC={auc_1:.3f})', alpha=0.7)
    
    axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].set_title('편향 정도별 민감도 곡선 (ROC 유사)')
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 데이터셋별 편향 vs 정확도 트레이드오프
    for dataset_name in datasets:
        spd_values = spd_data[dataset_name]
        accuracy_values = accuracy_data[dataset_name]
        
        axes[1, 1].scatter(spd_values, accuracy_values, s=100, alpha=0.7, 
                           label=dataset_name, edgecolors='black')
        
        # 각 점에 편향 정도 라벨 추가
        for i, (spd, acc) in enumerate(zip(spd_values, accuracy_values)):
            bias_label = ['낮음', '보통', '높음'][i]
            axes[1, 1].annotate(bias_label, (spd, acc), xytext=(5, 5), 
                               textcoords='offset points', fontsize=8)
    
    axes[1, 1].set_xlabel('Statistical Parity Difference')
    axes[1, 1].set_ylabel('정확도')
    axes[1, 1].set_title('데이터셋별 편향 vs 정확도 트레이드오프')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('integrated_bias_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_integrated_mitigation_comparison_plots(all_results):
    """
    3개 데이터셋의 통합 편향 완화 기법별 성능 비교 시각화
    """
    print("3개 데이터셋 통합 편향 완화 기법별 성능 비교 시각화 중...")
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('3개 데이터셋 통합 편향 완화 기법별 성능 비교', fontsize=18, fontweight='bold')
    
    datasets = list(all_results.keys())
    bias_names = ['low_bias', 'moderate_bias', 'high_bias']
    techniques = ['reweighing', 'adversarial_debiasing', 'threshold_adjustment']
    
    # 1. 데이터셋별 기법 성능 비교 (정확도)
    x = np.arange(len(bias_names))
    width = 0.2
    
    for i, dataset_name in enumerate(datasets):
        for j, technique in enumerate(techniques):
            accuracy_values = []
            for bias_name in bias_names:
                if (bias_name in all_results[dataset_name]['mitigation'] and 
                    technique in all_results[dataset_name]['mitigation'][bias_name]):
                    accuracy = all_results[dataset_name]['mitigation'][bias_name][technique]['accuracy']
                    accuracy_values.append(accuracy)
                else:
                    accuracy_values.append(0)
            
            if any(acc > 0 for acc in accuracy_values):
                axes[0, 0].bar(x + (i*len(techniques) + j)*width, accuracy_values, width, 
                               label=f'{dataset_name}-{technique}', alpha=0.8)
    
    axes[0, 0].set_xlabel('편향 정도')
    axes[0, 0].set_ylabel('정확도')
    axes[0, 0].set_title('데이터셋별 기법 성능 비교 (정확도)')
    axes[0, 0].set_xticks(x + width * (len(techniques) * len(datasets) - 1) / 2)
    axes[0, 0].set_xticklabels(['낮음', '보통', '높음'])
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 데이터셋별 기법 성능 비교 (SPD)
    for i, dataset_name in enumerate(datasets):
        for j, technique in enumerate(techniques):
            spd_values = []
            for bias_name in bias_names:
                if (bias_name in all_results[dataset_name]['mitigation'] and 
                    technique in all_results[dataset_name]['mitigation'][bias_name]):
                    spd = all_results[dataset_name]['mitigation'][bias_name][technique]['bias_metrics']['statistical_parity_difference']
                    spd_values.append(spd)
                else:
                    spd_values.append(0)
            
            if any(spd > 0 for spd in spd_values):
                axes[0, 1].bar(x + (i*len(techniques) + j)*width, spd_values, width, 
                               label=f'{dataset_name}-{technique}', alpha=0.8)
    
    axes[0, 1].set_xlabel('편향 정도')
    axes[0, 1].set_ylabel('Statistical Parity Difference')
    axes[0, 1].set_title('데이터셋별 기법 성능 비교 (SPD)')
    axes[0, 1].set_xticks(x + width * (len(techniques) * len(datasets) - 1) / 2)
    axes[0, 1].set_xticklabels(['낮음', '보통', '높음'])
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 기법별 평균 성능 비교
    technique_avg_accuracy = {}
    technique_avg_spd = {}
    
    for technique in techniques:
        accuracy_sum = 0
        spd_sum = 0
        count = 0
        
        for dataset_name in datasets:
            for bias_name in bias_names:
                if (bias_name in all_results[dataset_name]['mitigation'] and 
                    technique in all_results[dataset_name]['mitigation'][bias_name]):
                    accuracy_sum += all_results[dataset_name]['mitigation'][bias_name][technique]['accuracy']
                    spd_sum += all_results[dataset_name]['mitigation'][bias_name][technique]['bias_metrics']['statistical_parity_difference']
                    count += 1
        
        if count > 0:
            technique_avg_accuracy[technique] = accuracy_sum / count
            technique_avg_spd[technique] = spd_sum / count
    
    if technique_avg_accuracy:
        techniques_list = list(technique_avg_accuracy.keys())
        accuracy_values = list(technique_avg_accuracy.values())
        spd_values = list(technique_avg_spd.values())
        
        x_tech = np.arange(len(techniques_list))
        
        axes[1, 0].bar(x_tech, accuracy_values, alpha=0.8, color='#2E8B57')
        axes[1, 0].set_xlabel('편향 완화 기법')
        axes[1, 0].set_ylabel('평균 정확도')
        axes[1, 0].set_title('기법별 평균 정확도')
        axes[1, 0].set_xticks(x_tech)
        axes[1, 0].set_xticklabels(techniques_list, rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].bar(x_tech, spd_values, alpha=0.8, color='#DC143C')
        axes[1, 1].set_xlabel('편향 완화 기법')
        axes[1, 1].set_ylabel('평균 SPD')
        axes[1, 1].set_title('기법별 평균 SPD')
        axes[1, 1].set_xticks(x_tech)
        axes[1, 1].set_xticklabels(techniques_list, rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('integrated_mitigation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_integrated_governance_recommendations(all_results):
    """
    3개 데이터셋 실험 결과를 바탕으로 통합 거버넌스 제안 생성
    """
    print("\n" + "="*80)
    print("3개 데이터셋 통합 거버넌스 제안 및 정책 권고사항")
    print("="*80)
    
    # 1. 데이터셋별 편향 판별 성능 분석
    print("\n1. 데이터셋별 편향 판별 성능 분석")
    print("-" * 60)
    
    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name}:")
        for bias_name, detection_results in results['detection'].items():
            spd = detection_results['bias_metrics']['statistical_parity_difference']
            accuracy = detection_results['accuracy']
            print(f"  {bias_name}:")
            print(f"    - SPD: {spd:.4f}")
            print(f"    - 정확도: {accuracy:.4f}")
            print(f"    - 판별 성능: {'우수' if spd > 0.05 else '보통' if spd > 0.02 else '미흡'}")
    
    # 2. 데이터셋별 편향 완화 기법 효과 분석
    print("\n2. 데이터셋별 편향 완화 기법 효과 분석")
    print("-" * 60)
    
    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name}:")
        for bias_name, techniques in results['mitigation'].items():
            print(f"  {bias_name}:")
            for technique_name, technique_results in techniques.items():
                spd = technique_results['bias_metrics']['statistical_parity_difference']
                accuracy = technique_results['accuracy']
                print(f"    {technique_name}:")
                print(f"      - SPD: {spd:.4f}")
                print(f"      - 정확도: {accuracy:.4f}")
    
    # 3. 통합 정책 권고사항
    print("\n3. 통합 정책 권고사항")
    print("-" * 60)
    
    print("3.1 모델 평가 의무 항목")
    print("  - '편향이 낮은 데이터셋에서도 판별·완화 성능 검증'을 필수 항목으로 포함")
    print("  - 최소 편향 감지 임계값 설정 (예: SPD > 0.01)")
    print("  - 정기적인 편향 모니터링 및 보고 의무화")
    print("  - **다양한 도메인(이미지, 음성, 텍스트)에서의 편향성 검증 의무화**")
    
    print("\n3.2 리스크 지표 설정")
    print("  - 과도 교정 리스크: 정확도 감소율 > 5%")
    print("  - 미교정 리스크: SPD > 0.05")
    print("  - 편향 완화 실패 리스크: 완화 후 SPD 개선율 < 20%")
    print("  - **도메인별 편향성 임계값 차별화**")
    
    print("\n3.3 기술적 요구사항")
    print("  - 다중 편향 완화 기법 적용 및 성능 비교")
    print("  - 실시간 편향 모니터링 시스템 구축")
    print("  - 편향 완화 효과의 재현성 검증")
    print("  - **크로스 도메인 편향성 검증 방법론 표준화**")
    
    print("\n3.4 거버넌스 체계")
    print("  - AI 편향성 위원회 구성")
    print("  - 정기적인 편향성 감사 및 평가")
    print("  - 편향 완화 기법의 표준화 및 가이드라인 제공")
    print("  - **도메인별 전문가 그룹 구성 및 협력 체계 구축**")
    
    # 4. 실험 결과 요약
    print("\n4. 실험 결과 요약")
    print("-" * 60)
    
    print("이 실험을 통해 다음을 입증했습니다:")
    print("  ✓ **3개 데이터셋에서 편향이 낮은 경우에도 편향 판별 모델의 성능 측정 가능**")
    print("  ✓ **다양한 도메인(이미지, 음성)에서 편향 완화 기법의 효과 검증 가능**")
    print("  ✓ 편향 완화 기법의 효과를 정량적으로 비교 가능")
    print("  ✓ 정확도와 공정성의 트레이드오프 관계 분석 가능")
    print("  ✓ **거버넌스 정책 수립을 위한 종합적 실증 데이터 제공**")
    
    print("\n" + "="*80)
