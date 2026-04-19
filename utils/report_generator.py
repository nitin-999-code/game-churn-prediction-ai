"""
Report Generator

Creates structured reports containing player behavior summary,
churn risk interpretation, and engagement recommendations.
Optionally supports PDF export.
"""

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

def generate_structured_report(player_idx, player_data, risk_prob, risk_level, recommendations):
    """
    Generates a structured dictionary report for a single player.
    """
    # 1. Player Behavior Summary
    summary = f"Player {player_idx} shows specific engagement patterns: "
    metrics = []
    if 'SessionsPerWeek' in player_data:
        metrics.append(f"{player_data['SessionsPerWeek']} sessions/week")
    if 'AvgSessionDurationMinutes' in player_data:
        metrics.append(f"{player_data['AvgSessionDurationMinutes']} min/session")
    if 'PlayTimeHours' in player_data:
        metrics.append(f"{player_data['PlayTimeHours']} total hours played")
    
    if metrics:
        summary += ", ".join(metrics) + "."
    else:
        summary += "Limited gameplay data available."
        
    # 2. Risk Interpretation
    if risk_level == "High":
        interpretation_prefix = "High risk of churn"
    elif risk_level == "Medium":
        interpretation_prefix = "Moderate risk of churn"
    else:
        interpretation_prefix = "Low risk of churn"
        
    risk_interpretation = f"{interpretation_prefix} detected. The model estimates a {risk_prob*100:.1f}% churn probability based on historical data."
    
    # Assembly
    report = {
        "Player Behavior Summary": summary,
        "Churn Risk Interpretation": risk_interpretation,
        "Engagement Recommendations": recommendations,
        "Supporting References": "Based on gaming engagement research and historical player retention behaviors.",
        "Ethical Disclaimer": "Recommendations are automated suggestions. AI predictions should be reviewed by human moderators before applying aggressive retention tactics."
    }
    
    return report

def generate_pdf_report(report_data):
    """
    Takes the structured report dictionary and generates a PDF.
    Returns the PDF as bytes.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Engagement Optimization Report")
    
    c.setFont("Helvetica", 12)
    y_pos = height - 90
    
    for key, value in report_data.items():
        # Title of section
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y_pos, key + ":")
        y_pos -= 20
        
        c.setFont("Helvetica", 11)
        if isinstance(value, list):
            for item in value:
                c.drawString(70, y_pos, f"• {item}")
                y_pos -= 20
        else:
            # Simple text wrap logic
            text_object = c.beginText(70, y_pos)
            words = value.split()
            line = ""
            for word in words:
                if len(line) + len(word) > 80:
                    text_object.textLine(line)
                    line = word + " "
                    y_pos -= 15
                else:
                    line += word + " "
            if line:
                text_object.textLine(line)
                y_pos -= 15
            
            c.drawText(text_object)
            y_pos -= 10
            
        y_pos -= 10
        if y_pos < 50:
            c.showPage()
            y_pos = height - 50
    
    c.save()
    buffer.seek(0)
    return buffer.getvalue()
