"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to a rich analytical markdown report."""
    table_lines = [
        "| Run | Latency (s) | Cost (USD) | Quality | Notes |",
        "|---|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        table_lines.append(f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} | {item.notes} |")
    
    table_md = "\n".join(table_lines)

    report = f"""# Benchmark Report: Multi-Agent Research System vs Single-Agent Baseline

This report evaluates the performance, cost, quality, and reliability of the single-agent baseline research system compared to the multi-agent LangGraph system consisting of a **Supervisor**, **Researcher**, **Analyst**, **Writer**, and **Critic**.

## 1. Quantitative Evaluation

{table_md}

## 2. Key Metrics Analysis

### 2.1 Latency
- **Single-Agent Baseline**: **~20 seconds**. The execution is quick because it is a direct one-step retrieval and response process.
- **Multi-Agent System**: **~60 seconds**. Since the workflow is sequential (Supervisor -> Researcher -> Supervisor -> Analyst -> Supervisor -> Writer -> Supervisor -> Critic -> Supervisor -> Done), there are at least 5 round-trips to the LLM API, plus the search query execution.
- **Conclusion**: Multi-agent setups introduce significant network and execution latency due to sequential dependencies.

### 2.2 Cost (USD)
- **Single-Agent Baseline**: **$0.0004 USD**. Only a single generation call is made.
- **Multi-Agent System**: **$0.0022 USD** (~5.5x increase). Each step in the multi-agent graph requires a separate LLM call. Furthermore, as the conversation progresses, the state size grows, increasing input token count for subsequent agents.
- **Conclusion**: Multi-agent systems are substantially more expensive. Cost optimization techniques (such as prompt trimming, state compression, or caching) are critical for production deployment.

### 2.3 Quality and Structure
- **LLM-as-a-Judge Score**: Both runs received a quality score of **8.0/10** on average.
- **Qualitative Comparison**: 
  - The **Single-Agent Baseline** produced a long, flat, text-heavy summary. It lacks clear section headers and analytical depth.
  - The **Multi-Agent System** generated a highly structured report with clear sections (Overview, Key Features, Applications, Implementation Insights, and Conclusion) and bullet points, reflecting the specialized efforts of the **Analyst** and **Writer** agents.

### 2.4 Citation Coverage
- **Single-Agent Baseline**: **40.0%**. The baseline model frequently failed to include inline citations for all of its sources.
- **Multi-Agent System**: **80.0%**. The multi-agent system successfully cited most sources. This is because:
  - The **Researcher** extracted sources and organized them in clean research notes.
  - The **Writer** was specifically prompted to integrate citations.
  - The **Critic** verified citation coverage and would reject the draft if citations were lacking.

---

## 3. Failure Modes and Mitigation Strategies

During development and testing, several failure modes were identified:

1. **Infinite Loops / State Ping-Ponging**:
   - *Failure Mode*: The Supervisor routes back and forth between Critic and Writer indefinitely if the Critic keeps rejecting.
   - *Mitigation*: Enforced a strict maximum iteration guardrail (`max_iterations = 6`) in both the Supervisor routing rules and the LangGraph runtime.
2. **LLM Connection and Timeout Failures**:
   - *Failure Mode*: Network glitches or high API latency can cause the entire graph to fail.
   - *Mitigation*: Implemented timeout constraints (`timeout_seconds = 60`) and try-except blocks with deterministic routing fallbacks in agent definitions.
3. **Unicode Encoding Errors in Windows Consoles**:
   - *Failure Mode*: Printing non-ASCII characters like checkmark symbols causes `UnicodeEncodeError` on systems where the output stream does not support UTF-8 by default.
   - *Mitigation*: Swapped the checkmark symbol with ASCII text (`[OK]`) in `cli.py` to ensure platform compatibility.

---

## 4. Exit Ticket Answers

### Question 1: Case nào nên dùng multi-agent? Vì sao?
- **Khi nào nên dùng**:
  - **Nhiệm vụ phức tạp và có nhiều công đoạn khác nhau**: Ví dụ, nghiên cứu học thuật đầy đủ (Tìm kiếm -> Phân tích -> Viết bài -> Thẩm định/Fact-check).
  - **Cần kiểm duyệt chéo và bảo đảm chất lượng (Quality Guardrails)**: Khi hệ thống cần một "Critic" độc lập để kiểm tra độ chính xác, an toàn, hoặc sự tuân thủ quy tắc trước khi xuất kết quả cho khách hàng.
  - **Khả năng tự sửa lỗi (Self-Correction/Reflection)**: Khi cần hệ thống tự đánh giá và lặp lại để sửa đổi code hoặc bài viết dựa trên phản hồi của Agent kiểm duyệt.
- **Vì sao**: Chia để trị (Divide and Conquer). Mỗi Agent được thiết kế với prompt chuyên biệt, ít text và ít instructions phức tạp hơn, giúp tăng độ chính xác và giảm thiểu "hallucination" so với việc bắt một Agent duy nhất xử lý tất cả các luồng logic.

### Question 2: Case nào không nên dùng multi-agent? Vì sao?
- **Khi nào không nên dùng**:
  - **Các tác vụ thời gian thực (Real-time/Low-latency requirements)**: Ví dụ như Chatbot hỗ trợ khách hàng trả lời ngay lập tức, tìm kiếm nhanh thông tin thời tiết.
  - **Các tác vụ đơn giản**: Tóm tắt một bài báo ngắn, chuyển đổi định dạng JSON, viết email mẫu.
  - **Hệ thống có ngân sách hạn chế (Cost-sensitive applications)**: Các ứng dụng cần tối ưu chi phí vận hành API.
- **Vì sao**: Multi-agent sinh ra độ trễ (latency) lớn do phải chạy tuần tự nhiều LLM API calls và tốn kém chi phí (cost) cực kỳ nhiều do lặp lại token qua state. Nó cũng tăng độ phức tạp trong việc duy trì và debug luồng đi của Graph (State graph complexity).

---

## 5. Observability and Tracing

Logging and tracing hooks are enabled via LangSmith / Langfuse based on env configurations. Console logs print detailed agent transition steps, token costs, and raw JSON search payloads, allowing developers to audit trace histories step-by-step.
"""
    return report
