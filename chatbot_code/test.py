# test_v4.py
from func.sp_predict import predict_stock_price

result = predict_stock_price("삼성전자")

if result:
    cv = result['cv_summary']
    print(f"\n{'='*50}")
    print(f" {result['symbol']} ({result['stock_name']})")
    print(f"{'='*50}")
    print(f"  현재가:   {result['last_close']:>12,.2f}")
    print(f"  예측가:   {result['predicted_price']:>12,.2f}")
    print(f"  등락률:   {result['predicted_return']:>+11.4%}")
    print(f"  방향:     {result['direction']}")
    print(f"  시그널:   {result['signal']}")
    print(f"{'='*50}")
    print(f"  방향성:   {cv['Direction_Accuracy']['mean']:.2%} ± {cv['Direction_Accuracy']['std']:.2%}")
    print(f"  MAE:      {cv['MAE']['mean']:.6f}")
    print(f"  피처:     {len(result['selected_features'])}개")
    print(f"{'='*50}")