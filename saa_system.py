import pandas as pd
import numpy as np
import streamlit as st
from scipy.stats import chisquare # 벤포드의 법칙 통계 검증을 위함

# ==============================================================================
# 0. 환경 설정 및 모의 데이터 (Mock Data) 생성
#    - 실제 구현 시에는 이 부분을 DB 연결 및 API 호출 코드로 대체해야 합니다.
# ==============================================================================

# 룰 관리 테이블 (모듈 3에서 UI를 통해 수정 가능)
AUDIT_RULES = {
    'benford_alpha': 0.05,  # 벤포드 법칙 검증의 유의 수준 (p-value)
    'vendor_trend_threshold': 0.20,  # 전월 대비 비용 급증 임계치 (20%)
    'round_amount_threshold': 500000, # 딱 떨어지는 금액 탐지 기준 (50만원)
    'large_expense_limit': 10000000  # 내부 결재 규정 (1,000만원 초과 시 감사팀 경유)
}

@st.cache_data
def generate_mock_data():
    """모의 거래 데이터를 생성합니다. (실제 DB 연결 대체)"""
    np.random.seed(42)
    n_records = 5000
    
    # 정상 데이터 (벤포드 법칙을 따르는 경향)
    leading_digits = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9], size=n_records, p=[0.301, 0.176, 0.125, 0.097, 0.079, 0.067, 0.058, 0.051, 0.046])
    amounts = leading_digits * (10 ** np.random.uniform(3, 7, n_records))
    
    # 이상 데이터 주입 (벤포드 법칙 위반, 라운드 금액 등)
    # 100건을 '9'로 시작하도록 조작
    amounts[:100] = 9 * (10 ** np.random.uniform(3, 7, 100))
    # 50건을 딱 떨어지는 금액으로 조작
    amounts[100:150] = np.random.choice([1000000, 500000, 2000000], size=50)

    # DataFrame 생성
    data = {
        '거래일자': pd.to_datetime('2024-01-01') + pd.to_timedelta(np.random.randint(0, 300, n_records), unit='D'),
        '계정코드': np.random.choice(['4110_매출', '5110_급여', '5120_복리후생', '6210_접대비', '1310_가지급금'], n_records, p=[0.4, 0.2, 0.1, 0.1, 0.2]),
        '거래처명': [f'Vendor_{i}' for i in np.random.randint(1, 100, n_records)],
        '거래금액': amounts.round(0),
        '결재상태': np.random.choice(['승인', '미승인'], n_records, p=[0.95, 0.05]),
        '증빙여부': np.random.choice([True, False], n_records, p=[0.99, 0.01])
    }
    df = pd.DataFrame(data)
    
    # 1000만원 초과 지출 건에 결재 오류 주입
    df.loc[(df['거래금액'] > AUDIT_RULES['large_expense_limit']) & (df['결재상태'] == '승인'), '결재상태'] = np.random.choice(['승인', '미승인_감사규정위반'], size=df[(df['거래금액'] > AUDIT_RULES['large_expense_limit']) & (df['결재상태'] == '승인')].shape[0], p=[0.9, 0.1])
    
    return df

# ==============================================================================
# 1. 모듈 1: 데이터 연동 및 정제 모듈 (Data Integration & Cleaning)
# ==============================================================================

def fetch_data(source='Mock_ERP_DB'):
    """데이터 소스에서 데이터를 추출합니다."""
    st.info(f"💾 데이터 소스 '{source}'에서 데이터 추출을 시작합니다.")
    return generate_mock_data()

def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """데이터 유효성 검증 및 정규화를 수행합니다."""
    # 1. 유효성 검증: 거래금액 0원 이하, 결측치 등 간단 체크
    invalid_rows = df[df['거래금액'] <= 0]
    if not invalid_rows.empty:
        st.warning(f"⚠️ 경고: 0원 이하 거래 {len(invalid_rows)}건이 발견되었습니다. 분석에서 제외합니다.")
        df = df[df['거래금액'] > 0]
        
    # 2. 데이터 형식 통일
    df['거래일자'] = pd.to_datetime(df['거래일자'])
    df['거래금액'] = df['거래금액'].astype(float)
    
    # 3. 거래처명 정규화 (LLM 활용 필요 시)
    df['거래처명_정규화'] = df['거래처명'].str.replace(r'\(주\)|\(유\)', '', regex=True).str.strip()
    
    st.success(f"✅ 데이터 정규화 완료. 총 {len(df)}건의 유효 데이터를 분석에 사용합니다.")
    return df

# ==============================================================================
# 2. 모듈 2: 핵심 리스크 상시 탐지 모듈 (Core Risk Continuous Detection)
# ==============================================================================

def detect_benford_anomaly(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """
    재무 리스크: 벤포드의 법칙을 이용한 매출/비용 데이터 조작 탐지
    """
    # 거래금액의 첫째 자리 숫자를 추출
    df['선행숫자'] = df['거래금액'].astype(str).str[0].astype(int)
    
    # 벤포드 법칙 기대 확률
    benford_probs = np.log10(1 + 1 / np.arange(1, 10))
    expected_counts = benford_probs * len(df)
    
    # 실제 빈도 계산
    actual_counts = df['선행숫자'].value_counts().sort_index().reindex(np.arange(1, 10), fill_value=0)
    
    # 카이제곱 검정
    if any(expected_counts < 5):
        st.warning("경고: 기대 빈도가 낮아 카이제곱 검정 결과의 신뢰도가 떨어질 수 있습니다.")
        
    chi2_stat, p_value = chisquare(actual_counts, expected_counts)
    
    result_df = pd.DataFrame({
        '선행숫자': np.arange(1, 10),
        '기대빈도(%)': (benford_probs * 100).round(2),
        '실제빈도(%)': (actual_counts / len(df) * 100).round(2)
    })
    
    is_anomaly = p_value < alpha
    st.info(f"🔍 벤포드 법칙 검정 결과: Chi2={chi2_stat:.2f}, P-value={p_value:.4f}")
    
    return result_df, is_anomaly, p_value

def analyze_vendor_trend(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    재무 리스크: 거래처별 비용 급증 탐지
    """
    # 급여, 복리후생, 접대비 등 비용 계정만 필터링
    cost_df = df[df['계정코드'].str.contains('5|6')]
    
    # 월별 거래처/계정별 집계
    cost_df['YearMonth'] = cost_df['거래일자'].dt.to_period('M')
    monthly_summary = cost_df.groupby(['YearMonth', '거래처명_정규화'])['거래금액'].sum().reset_index()
    
    # 전월 대비 변동률 계산
    monthly_summary['Prev_Month_Amount'] = monthly_summary.groupby('거래처명_정규화')['거래금액'].shift(1)
    monthly_summary['Change_Rate'] = (monthly_summary['거래금액'] - monthly_summary['Prev_Month_Amount']) / monthly_summary['Prev_Month_Amount']
    
    # 임계치 초과 건 탐지
    anomalies = monthly_summary[
        (monthly_summary['Change_Rate'].abs() > threshold) & 
        (monthly_summary['Prev_Month_Amount'].notna()) &
        (monthly_summary['거래금액'] > 0)
    ].sort_values('Change_Rate', ascending=False)
    
    return anomalies[['YearMonth', '거래처명_정규화', '거래금액', 'Prev_Month_Amount', 'Change_Rate']]

def check_approval_violation(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    """
    규정 준수 리스크: 내부 결재 규정 위반 탐지 (1,000만원 초과 지출 시 감사 규정 위반 등)
    """
    # 1. 금액 임계치 초과 지출 필터링
    large_expenses = df[df['거래금액'] > limit].copy()
    
    # 2. 미승인 상태 확인 (예: '미승인_감사규정위반' 상태를 위반으로 간주)
    violations = large_expenses[large_expenses['결재상태'].str.contains('미승인')]
    
    return violations[['거래일자', '계정코드', '거래처명', '거래금액', '결재상태']]

def mock_llm_analyze_contract(contract_text: str, law_list: list) -> dict:
    """
    규정 준수 리스크: LLM을 이용한 계약서 핵심 조항 검토 (가상 함수)
    """
    st.info("🤖 LLM API를 호출하여 계약서 핵심 조항 분석 중... (가상 실행)")
    
    if "하도급" in contract_text:
        return {
            "is_compliant": False,
            "score": 65, # 낮은 점수
            "findings": ["하도급 대금 지급 기일에 대한 명확한 명시가 부족합니다.", "기술 자료 보호 조항이 일반적인 수준에 그쳐 회사에 불리할 수 있습니다."],
            "summary": "하도급법 관련 필수 조항 누락 및 법적 리스크 존재 가능성이 탐지됨."
        }
    else:
        return {
            "is_compliant": True,
            "score": 90,
            "findings": ["특이 사항 없음. 일반적인 상거래 규정을 준수합니다."],
            "summary": "계약서 컴플라이언스 위험도가 낮습니다."
        }


# ==============================================================================
# 3. 모듈 3: 자동 보고 및 알림 모듈 (Automated Reporting & Notification)
# ==============================================================================

def generate_report_summary(anomalies: pd.DataFrame, benford_anomaly: bool, p_value: float, rule_violations: pd.DataFrame, contract_result: dict):
    """
    탐지된 핵심 리스크를 기반으로 경영진 보고서 초안을 LLM을 통해 자동 생성합니다. (가상 함수)
    """
    st.subheader("📊 자동 보고서 초안 생성 결과 (LLM 기반)")
    
    summary = f"""
    ## [SAA] 핵심 리스크 요약 보고 (1인 감사조직용)

    ### 1. 주요 리스크 탐지 현황
    | 리스크 유형 | 탐지 여부 | 상세 내용 |
    | :--- | :--- | :--- |
    | **재무 리스크 (벤포드)** | {'🚨 위험' if benford_anomaly else '✅ 정상'} | 선행 숫자 분포 P-value: {p_value:.4f} ({'유의수준 이하로 조작 의심 패턴 탐지' if benford_anomaly else '정상 범위'}) |
    | **비용 급증 (Vendor Trend)** | {'🚨 위험' if not anomalies.empty else '✅ 정상'} | 전월 대비 {len(anomalies)}건의 거래에서 임계치({st.session_state.rules['vendor_trend_threshold']*100}%)를 초과하는 급증 패턴 탐지 |
    | **내부 규정 위반** | {'🚨 위험' if not rule_violations.empty else '✅ 정상'} | 총 {len(rule_violations)}건의 지출에서 1천만원 초과 건에 대한 결재/승인 규정 위반 의심 |
    | **계약서 컴플라이언스** | {'🚨 위험' if not contract_result['is_compliant'] else '✅ 정상'} | 컴플라이언스 점수 {contract_result['score']}점. 주요 발견사항: {', '.join(contract_result['findings'][:1])}... |

    ### 2. 감사 처분 요구서 초안 (권고 사항)
    1. **벤포드 리스크:** 선행 숫자 9의 과도한 집중 현상에 대해 해당 계정(매출 또는 비용)의 **원본 증빙 자료**를 검토하고 **재무 기록의 무결성**을 확보할 것을 권고함.
    2. **비용 급증:** 변동률이 가장 높은 거래처 ({anomalies['거래처명_정규화'].iloc[0] if not anomalies.empty else 'N/A'})와의 거래에 대해 **허위 증빙 여부**를 확인하고, **예산 통제 프로세스**를 강화해야 함.
    3. **내부 통제:** {len(rule_violations)}건의 규정 위반 건에 대해 해당 결재라인의 **책임 소재를 명확히** 하고 재발 방지 교육을 즉시 시행할 것을 권고함.
    """
    
    st.markdown(summary)
    
    st.download_button(
        label="📄 보고서 초안 다운로드 (Mock)",
        data=summary,
        file_name="SAA_Audit_Report_Summary.md",
        mime="text/markdown"
    )

def send_alert(alert_message: str):
    """
    이메일 또는 메신저로 리스크 알림을 발송합니다. (가상 함수)
    """
    st.sidebar.error(f"🚨 **리스크 즉시 알림 발송:** {alert_message}")

# ==============================================================================
# 4. Streamlit 기반 SAA 시스템 UI (메인 함수)
# ==============================================================================

def saa_main():
    """SAA 시스템의 메인 인터페이스입니다."""
    st.set_page_config(layout="wide")
    st.title("🛡️ 1인 감사 조직용 SAA (System-Assisted Audit) 시스템")
    st.markdown("---")

    # 사이드바: 룰 관리 탭 (Streamlit Session State 사용)
    st.sidebar.header("🛠️ 감사 규칙 관리")
    
    global AUDIT_RULES
    if 'rules' not in st.session_state:
        st.session_state.rules = AUDIT_RULES

    # UI를 통해 룰 수정 가능하도록 설정
    st.session_state.rules['benford_alpha'] = st.sidebar.slider(
        '벤포드 P-value 임계치', 0.01, 0.10, st.session_state.rules['benford_alpha'], 0.005
    )
    st.session_state.rules['vendor_trend_threshold'] = st.sidebar.slider(
        '비용 급증 변동률 임계치', 0.05, 0.50, st.session_state.rules['vendor_trend_threshold'], 0.01
    )
    st.session_state.rules['large_expense_limit'] = st.sidebar.number_input(
        '대형 지출 규정 금액', 5000000, 50000000, st.session_state.rules['large_expense_limit'], 1000000
    )
    
    if st.sidebar.button("규칙 적용 및 감사 재실행"):
        st.cache_data.clear()
        st.success("새로운 규칙이 적용되었습니다. 감사를 재실행합니다.")
        

    # 메인 영역
    if st.button("🚀 SAA 감사 실행"):
        
        # 1단계: 데이터 연동 및 정제
        df = fetch_data()
        df_clean = normalize_data(df)

        st.markdown("## 1. ⚙️ 데이터 연동 및 정제 완료")
        st.dataframe(df_clean.head(), use_container_width=True)
        st.markdown("---")
        
        # 2단계: 핵심 리스크 상시 탐지
        st.markdown("## 2. 🛡️ 핵심 리스크 상시 탐지 모듈 실행")
        
        col1, col2 = st.columns(2)

        # 2-1. 재무 리스크: 벤포드 법칙
        with col1:
            st.subheader("2-1. 벤포드 법칙 기반 부정 탐지")
            benford_df, is_benford_anomaly, p_value = detect_benford_anomaly(
                df_clean[df_clean['계정코드'].str.contains('매출')], st.session_state.rules['benford_alpha']
            )
            
            if is_benford_anomaly:
                st.error("🚨 **위험 탐지:** 매출 데이터에서 통계적 이상 패턴(조작 의심)이 탐지되었습니다.")
                send_alert(f"매출 벤포드 P-value {p_value:.4f} (임계치 {st.session_state.rules['benford_alpha']} 이하)")
            else:
                st.success("✅ 벤포드 법칙: 정상 범위입니다.")
            
            st.dataframe(benford_df, use_container_width=True)
            st.bar_chart(benford_df.set_index('선행숫자'))
            
        # 2-2. 재무 리스크: 비용 급증 탐지
        with col2:
            st.subheader("2-2. 거래처별 비용 급증 패턴 탐지")
            trend_anomalies = analyze_vendor_trend(df_clean, st.session_state.rules['vendor_trend_threshold'])
            
            if not trend_anomalies.empty:
                st.warning(f"⚠️ **이상 탐지:** 총 {len(trend_anomalies)}건의 거래처에서 비용 급증이 감지되었습니다.")
                send_alert(f"비용 급증: {trend_anomalies.iloc[0]['거래처명_정규화']} 등 {len(trend_anomalies)}건")
            else:
                st.success("✅ 비용 급증: 특이 사항 없음.")
                
            st.dataframe(trend_anomalies.head(10), use_container_width=True)

        st.markdown("---")

        # 2-3. 규정 준수 리스크: 내부 규정 위반
        st.subheader("2-3. 내부 결재 규정 위반 탐지")
        rule_violations = check_approval_violation(df_clean, st.session_state.rules['large_expense_limit'])
        
        if not rule_violations.empty:
            st.error(f"🚨 **규정 위반:** 총 {len(rule_violations)}건에서 대형 지출 결재 규정 위반이 감지되었습니다.")
            send_alert(f"내부 규정 위반: 1천만원 초과 결재 오류 {len(rule_violations)}건")
        else:
            st.success("✅ 내부 규정 준수: 특이 사항 없음.")
            
        st.dataframe(rule_violations, use_container_width=True)
        
        st.markdown("---")

        # 2-4. 규정 준수 리스크: 계약서 컴플라이언스 (LLM Mock)
        st.subheader("2-4. LLM 기반 계약서 컴플라이언스 분석")
        
        # 모의 계약서 데이터 
        mock_contract_text = "이 계약은 A사와 B사의 하도급 거래에 관한 것이며, 대금은 90일 이내에 지급한다. 기술 자료 보호에 대한 조항은 일반적인 수준을 따른다."
        law_list = ["하도급법", "공정거래법"]
        
        contract_result = mock_llm_analyze_contract(mock_contract_text, law_list)
        
        if not contract_result['is_compliant']:
            st.error(f"🚨 **계약서 리스크:** {contract_result['summary']}")
        else:
            st.success(f"✅ 계약서 분석: {contract_result['summary']}")
        
        st.json(contract_result)

        st.markdown("---")

        # 3단계: 자동 보고 및 알림
        st.markdown("## 3. 📄 자동 보고 및 알림 모듈 실행")
        generate_report_summary(trend_anomalies, is_benford_anomaly, p_value, rule_violations, contract_result)


if __name__ == '__main__':
    saa_main()