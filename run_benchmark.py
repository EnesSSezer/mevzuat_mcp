import asyncio
import json
import logging
import os
import random
import time
import re
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

# .env dosyasındaki ortam değişkenlerini yükle
load_dotenv()

from mcp_client.agent import Agent
from mcp_client.config import ClientConfig
from mcp_client.events import AgentEvent, EventType
from mcp_client.state import ConversationState

# Log seviyesini ayarla
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("mcp_benchmark")

# ---------------------------------------------------------------------------
# BOT KORUMASI İÇİN BEKLEME SÜRESİ AYARLARI (Saniye)
# ---------------------------------------------------------------------------
MIN_DELAY_SEC = 3.0  # Minimum bekleme süresi
MAX_DELAY_SEC = 7.0  # Maksimum bekleme süresi

# ---------------------------------------------------------------------------
# 1. GEMINI EVALUATOR (LLM-as-a-Judge) YAPILANDIRMASI
# ---------------------------------------------------------------------------
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if not gemini_api_key:
    logger.warning("GEMINI_API_KEY ortam değişkeni ayarlanmamış. Evaluator çağrıları başarısız olabilir.")

client = OpenAI(
    api_key=gemini_api_key or "dummy_key",  # Your Gemini API Key
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

EVALUATOR_SYSTEM_PROMPT = """
Sen uzman bir Türk Hukuk akademisyeni ve Yapay Zeka Değerlendirme Uzmanısın.
Sana bir kullanıcı sorusu, bu sorunun kanunen OLMASI GEREKEN ALTIN YANITI (Gold Answer),
değerlendirme kriterleri ve bir AI Asistanının ürettiği yanıt verilecek.

Görevin:
1. AI Asistanının yanıtını Altın Yanıt ve Kriterler ile karşılaştır.
2. Yanıtta hukuki halüsinasyon, yanlış madde atfı, yanlış süre bilgisi veya mevzuat karıştırma var mı kontrol et.
3. 0-100 arasında bir puan ver (60 ve üzeri PASS kabul edilir).
4. Sadece aşağıdaki JSON formatında yanıt dön:

{
  "score": int,
  "status": "PASS" veya "FAIL",
  "matched_criteria": ["karşılanan kriter 1", "karşılanan kriter 2"],
  "missing_criteria": ["kaçırılan kriter 1"],
  "hallucination_detected": bool,
  "reasoning": "Kısa ve net değerlendirme açıklaması"
}
"""

def _clean_and_parse_json(text: str) -> Dict[str, Any]:
    """Extacts and parses JSON dict from LLM response robustly."""
    text = text.strip()
    # Strip markdown blocks if present
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    # Try parsing directly first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
        
    # Fallback: remove trailing commas before closing braces/brackets
    text_no_comma = re.sub(r",\s*([\}\]])", r"\1", text)
    try:
        return json.loads(text_no_comma)
    except json.JSONDecodeError:
        pass
        
    # Fallback 2: Extract from first { to last } (but avoid matching garbage after if it has braces)
    match = re.search(r"\{.*\}", text_no_comma, re.DOTALL)
    if match:
        json_str = match.group(0).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
            
    # If all fails, return a safe fallback dict
    return {
        "score": 0,
        "status": "FAIL",
        "reasoning": f"JSON parsing failed. Raw output: {text[:100]}...",
        "hallucination_detected": False
    }

def evaluate_with_gemini(question: str, gold_answer: str, criteria: List[str], agent_response: str) -> Dict[str, Any]:
    """Gemini API kullanarak agent yanıtını puanlar."""
    prompt = f"""
    --- KULLANICI SORUSU ---
    {question}

    --- ALTIN YANIT (GROUND TRUTH) ---
    {gold_answer}

    --- DEĞERLENDİRME KRİTERLERİ ---
    {json.dumps(criteria, ensure_ascii=False, indent=2)}

    --- AI ASİSTANIN VERDİĞİ YANIT ---
    {agent_response}
    """

    try:
        response = client.chat.completions.create(
            model="gemini-3.5-flash-lite",
            messages=[
                {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        content = response.choices[0].message.content
        return _clean_and_parse_json(content)
    except Exception as e:
        logger.error(f"Gemini Evaluator Hatalı: {e}")
        return {
            "score": 0,
            "status": "FAIL",
            "reasoning": f"Gemini Evaluator hatası: {str(e)}",
            "hallucination_detected": False
        }

# ---------------------------------------------------------------------------
# 2. BENCHMARK OTOMASYONU
# ---------------------------------------------------------------------------
async def run_benchmark(benchmark_file: str = "mevzuat_mcp_benchmark.json"):
    if not os.path.exists(benchmark_file):
        print(f"❌ HATA: '{benchmark_file}' dosyası bulunamadı!")
        return

    with open(benchmark_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # Agent Yapılandırması
    config = ClientConfig()
    
    # Anlık tool ve loop guard sayaçları (Soru başına sıfırlanır)
    current_turn_tool_calls = 0
    loop_guard_confirmations_for_current_question = 0

    async def benchmark_confirm_loop_guard(reason: str) -> bool:
        nonlocal loop_guard_confirmations_for_current_question
        loop_guard_confirmations_for_current_question += 1
        if loop_guard_confirmations_for_current_question <= 1:
            logger.warning(
                f"Loop guard tetiklendi #{loop_guard_confirmations_for_current_question}: {reason} -> 1 kez onay verildi."
            )
            print(f"   ⚠️  Loop Guard: {reason} -> Otomatik onay verildi (1/1).")
            return True
        else:
            logger.warning(
                f"Loop guard tetiklendi #{loop_guard_confirmations_for_current_question}: {reason} -> 2. kez tetiklendi, reddediliyor."
            )
            print(f"   ⛔ Loop Guard: {reason} -> 2. kez tetiklendi, döngüyü durdurmak için reddedildi.")
            return False

    async def benchmark_event_handler(event: AgentEvent) -> None:
        nonlocal current_turn_tool_calls
        if event.type == EventType.TOOL_CALL_STARTED:
            current_turn_tool_calls += 1
            print(f"   🛠️  Tool Çağrıldı: {event.tool_name}")
        elif event.type == EventType.LOOP_GUARD_TRIGGERED:
            print(f"   ⚠️  Loop Guard Uyarısı: {event.detail}")

    agent = Agent(
        config=config,
        confirm_callback=benchmark_confirm_loop_guard,
        on_event=benchmark_event_handler
    )

    print("🔌 MCP Sunucusuna bağlanılıyor...")
    await agent.start()
    print(f"✅ Bağlantı başarılı! {len(agent.transport.tools)} tool aktif.")
    print(f"🚀 Mevzuat MCP Benchmark Testi Başlatılıyor... ({len(questions)} Soru)\n" + "="*70)

    results = []
    total_start_time = time.time()
    total_tool_calls = 0
    total_score = 0
    passed_count = 0

    try:
        for idx, item in enumerate(questions, 1):
            q_id = item["id"]
            question = item["question"]
            level = item["difficulty_level"]
            
            print(f"\n▶ [{idx}/{len(questions)}] Soru ID: {q_id} (Level {level})")
            print(f"❓ Soru: {question}")

            # 1. Oturum durumunu ve loop guard'ı her soru öncesi temizle (Bağlam Karışmasını Önler)
            system_prompt = agent.state.messages[0]["content"]
            agent.state = ConversationState(system_prompt)
            agent.loop_guard.reset()
            agent._round_index = 0
            current_turn_tool_calls = 0
            loop_guard_confirmations_for_current_question = 0

            # 2. Agent'ı Çalıştır ve Süreyi Ölç
            q_start_time = time.time()
            agent_response = await agent.run_turn(question)
            latency_sec = round(time.time() - q_start_time, 2)
            
            tool_calls_made = current_turn_tool_calls
            total_tool_calls += tool_calls_made

            print(f"💬 Yanıt Alındı ({latency_sec}s | {tool_calls_made} Tool Çağrısı)")
            print("🔍 Gemini Evaluator ile doğrulama yapılıyor...")

            # 3. Gemini API ile Puanla
            eval_result = evaluate_with_gemini(
                question=question,
                gold_answer=item["gold_answer"],
                criteria=item["evaluation_criteria"],
                agent_response=agent_response
            )

            score = eval_result.get("score", 0)
            status = eval_result.get("status", "FAIL")
            total_score += score
            if status == "PASS":
                passed_count += 1

            # Anlık Sonuç Ekranı
            status_icon = "✅ PASS" if status == "PASS" else "❌ FAIL"
            print(f"📊 Skor: {score}/100 [{status_icon}] | Halüsinasyon: {eval_result.get('hallucination_detected')}")
            print(f"📝 Değerlendirme: {eval_result.get('reasoning')}")

            # Detaylı kayıt
            results.append({
                "id": q_id,
                "level": level,
                "category": item.get("category"),
                "question": question,
                "agent_response": agent_response,
                "tool_calls": tool_calls_made,
                "latency_sec": latency_sec,
                "evaluation": eval_result
            })

            # 4. Değişken Timeout (Bot korumasını engellemek için)
            if idx < len(questions):
                delay = round(random.uniform(MIN_DELAY_SEC, MAX_DELAY_SEC), 1)
                print(f"⏳ Bot engeline takılmamak için {delay} saniye bekleniyor...")
                await asyncio.sleep(delay)

    finally:
        await agent.shutdown()
        print("\n🔌 MCP Sunucu bağlantısı kapatıldı.")

    total_execution_time = round(time.time() - total_start_time, 2)
    avg_score = round(total_score / len(questions), 2)

    # ---------------------------------------------------------------------------
    # 3. ÖZET VE RAPORLAMA
    # ---------------------------------------------------------------------------
    summary = {
        "total_questions": len(questions),
        "passed_questions": passed_count,
        "success_rate_percent": round((passed_count / len(questions)) * 100, 2),
        "average_score": avg_score,
        "total_execution_time_sec": total_execution_time,
        "total_tool_calls": total_tool_calls,
        "avg_tool_calls_per_query": round(total_tool_calls / len(questions), 2)
    }

    report = {
        "summary": summary,
        "detailed_results": results
    }

    report_filename = "benchmark_results_report.json"
    with open(report_filename, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "="*70)
    print("🏆 BENCHMARK TESTİ TAMAMLANDI!")
    print(f"⏱️  Toplam Süre: {total_execution_time} saniye")
    print(f"🛠️  Toplam Tool Çağrısı: {total_tool_calls} (Soru başına ort. {summary['avg_tool_calls_per_query']})")
    print(f"🎯 Genel Başarı Oranı: %{summary['success_rate_percent']} ({passed_count}/{len(questions)} PASS)")
    print(f"⭐ Ortalama Puan: {avg_score} / 100")
    print(f"📄 Detaylı rapor '{report_filename}' dosyasına kaydedildi.")

if __name__ == "__main__":
    asyncio.run(run_benchmark())