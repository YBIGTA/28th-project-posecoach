"""
분석 결과를 사람이 읽기 쉬운 PDF로 변환한다.
"""
from collections import Counter
from datetime import datetime
from io import BytesIO
from pathlib import Path


def _resolve_fonts():
    # 한글 폰트가 있으면 사용하고, 없으면 기본 폰트로 폴백한다.
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    # (일반체 경로, 볼드체 경로) 쌍으로 관리
    candidates = [
        # Modal 컨테이너 (/root/assets/ 에 마운트)
        ("NotoSansKR", "/root/assets/NotoSansKR-Regular.ttf", "/root/assets/NotoSansKR-Bold.ttf"),
        # 로컬 개발 (프로젝트 상대 경로)
        ("NotoSansKR", str(Path(__file__).resolve().parents[1] / "apps/assets/NotoSansKR-Regular.ttf"),
                       str(Path(__file__).resolve().parents[1] / "apps/assets/NotoSansKR-Bold.ttf")),
        # 시스템 폰트 폴백
        ("AppleGothic", "/System/Library/Fonts/Supplemental/AppleGothic.ttf", None),
        ("NanumGothic", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf", None),
        ("MalgunGothic", r"C:\Windows\Fonts\malgun.ttf", None),
    ]
    for entry in candidates:
        font_name, regular_path, bold_path = entry
        if not Path(regular_path).exists():
            continue
        try:
            pdfmetrics.registerFont(TTFont(font_name, regular_path))
            bold_name = f"{font_name}-Bold"
            actual_bold = bold_path if bold_path and Path(bold_path).exists() else regular_path
            pdfmetrics.registerFont(TTFont(bold_name, actual_bold))
            return font_name, bold_name
        except Exception:
            continue

    return "Helvetica", "Helvetica-Bold"


def _format_percent(value):
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def _build_summary_rows(export_data):
    return [
        ["영상", str(export_data.get("video", "-"))],
        ["운동 종류", str(export_data.get("exercise_type", "-"))],
        ["운동 횟수", f"{int(export_data.get('exercise_count', 0))}회"],
        ["평균 자세 점수", _format_percent(export_data.get("avg_posture_score"))],
        ["오류 프레임", f"{int(export_data.get('error_frame_count', 0))}개"],
        ["총 프레임", f"{int(export_data.get('total_frames', 0))}개"],
        ["모션 선별 프레임", f"{int(export_data.get('analysis_target_frames', 0))}개"],
        ["실평가 프레임", f"{int(export_data.get('evaluated_frames', 0))}개"],
        ["키포인트 검출 성공", f"{int(export_data.get('extracted_keypoints', 0))}개"],
        ["FPS", f"{export_data.get('fps', '-') }"],
        ["해상도", "x".join(map(str, export_data.get("resolution", ["-", "-"])))],
    ]


def build_analysis_report_pdf(export_data, frame_scores, error_frames):
    """
    Args:
        export_data: app.py의 export_data dict
        frame_scores: 프레임 평가 결과 리스트
        error_frames: 오류 프레임 리스트
    Returns:
        bytes: PDF 바이너리
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    base_font, bold_font = _resolve_fonts()
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleKo",
        parent=styles["Title"],
        fontName=bold_font,
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1f2937"),
    )
    heading_style = ParagraphStyle(
        "HeadingKo",
        parent=styles["Heading2"],
        fontName=bold_font,
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#111827"),
    )
    normal_style = ParagraphStyle(
        "NormalKo",
        parent=styles["Normal"],
        fontName=base_font,
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#111827"),
    )

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title=f"{export_data.get('video', 'analysis')}_report",
    )
    story = []

    story.append(Paragraph("운동 분석 리포트", title_style))
    story.append(Spacer(1, 4 * mm))
    story.append(
        Paragraph(
            f"생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            normal_style,
        )
    )
    story.append(Spacer(1, 6 * mm))

    story.append(Paragraph("1) 요약", heading_style))
    summary_rows = _build_summary_rows(export_data)
    summary_table = Table(summary_rows, colWidths=[48 * mm, 118 * mm])
    summary_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), base_font),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f3f4f6")),
                ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#d1d5db")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 6 * mm))

    error_counter = Counter()
    for fs in frame_scores:
        for msg in fs.get("errors", []):
            if msg and msg != "키포인트 없음":
                error_counter[msg] += 1

    story.append(Paragraph("2) 주요 오류 요약", heading_style))
    error_rows = [["오류 메시지", "발생 횟수"]]
    if error_counter:
        for msg, cnt in error_counter.most_common(8):
            error_rows.append([msg, f"{cnt}회"])
    else:
        error_rows.append(["오류 없음", "0회"])

    error_table = Table(error_rows, colWidths=[130 * mm, 36 * mm])
    error_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), base_font),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(error_table)
    story.append(Spacer(1, 6 * mm))

    story.append(Paragraph("3) 오류 프레임 상세 (낮은 점수 순)", heading_style))
    frame_rows = [["프레임", "점수", "오류"]]
    for ef in sorted(error_frames, key=lambda x: x.get("score", 1.0))[:12]:
        frame_rows.append(
            [
                str(ef.get("frame_idx", "-")),
                _format_percent(ef.get("score")),
                ", ".join(ef.get("errors", [])[:2]) or "-",
            ]
        )
    if len(frame_rows) == 1:
        frame_rows.append(["-", "-", "기록된 오류 프레임이 없습니다."])

    frame_table = Table(frame_rows, colWidths=[20 * mm, 22 * mm, 124 * mm])
    frame_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), base_font),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(frame_table)
    story.append(Spacer(1, 6 * mm))

    score_text = _format_percent(export_data.get("avg_posture_score"))
    advice = "평균 점수가 낮은 경우, 오류 빈도 상위 항목부터 우선 교정하세요."
    story.append(Paragraph("4) 코멘트", heading_style))
    story.append(Paragraph(f"- 평균 자세 점수: {score_text}", normal_style))
    story.append(Paragraph(f"- 권장 조치: {advice}", normal_style))

    doc.build(story)
    return buffer.getvalue()