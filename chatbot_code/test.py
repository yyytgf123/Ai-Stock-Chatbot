# test_v6_1.py
from func.sp_predict import predict_stock_price

result = predict_stock_price("삼성전자", n_splits=5, optuna_trials=50, years=5)

if result:
    print("\n" + "=" * 60)
    print(" 최종 요약")
    print("=" * 60)
    print(f"  시그널:     {result['signal']}")
    print(f"  방향:       {result['direction']}")
    print(f"  Raw→보정:   {result['raw_probability']:.2%} → {result['calibrated_probability']:.2%}")
    print(f"  확신도:     {result['confidence']:.2%}")
    print(f"  예측가:     {result['predicted_price']:,.2f}")

    cv = result['cv_summary']
    print(f"\n  방향성 정확도: {cv['Direction_Accuracy']['mean']:.2%} ± {cv['Direction_Accuracy']['std']:.2%}")
    print(f"  사용 피처:     {len(result['selected_features'])}개")

    sentiment_feats = [f for f in result['selected_features']
                       if f in ['News_Sentiment', 'News_Positive_Ratio',
                                'News_Count', 'News_Sentiment_Std', 'News_Momentum']]
    print(f"  감성 피처:     {sentiment_feats if sentiment_feats else '없음 (중요도 미달)'}")

    if result.get('confidence_bins'):
        print("\n  확신도 구간별 적중률:")
        for label, info in result['confidence_bins'].items():
            if info['count'] > 0:
                print(f"    {label:>10s}: {info['accuracy']:.2%} ({info['count']}건)")