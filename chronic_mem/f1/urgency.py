"""
紧迫性标量 (Urgency Scalar) 检测器
检测对话中的情感与程度副词，量化临床紧迫性
"""
import re
from typing import List, Dict

class UrgencyDetector:
    """紧迫性检测器"""
    
    # 高危词汇（高紧迫性）
    HIGH_URGENCY_KEYWORDS = [
        "severe", "serious", "critical", "acute", "emergency",
        "urgent", "life-threatening", "dangerous", "intense",
        "剧烈", "严重", "抢救", "紧急", "危险", "急性", "致命"
    ]
    
    # 中等紧迫性词汇
    MEDIUM_URGENCY_KEYWORDS = [
        "moderate", "significant", "considerable", "substantial",
        "中等", "显著", "明显", "重要"
    ]
    
    # 低紧迫性词汇（常规）
    LOW_URGENCY_KEYWORDS = [
        "mild", "slight", "minor", "routine", "normal",
        "轻微", "轻度", "常规", "正常", "一般"
    ]
    
    def __init__(self):
        # 编译正则表达式以提高性能
        self.high_pattern = re.compile(
            r'\b(' + '|'.join(self.HIGH_URGENCY_KEYWORDS) + r')\b',
            re.IGNORECASE
        )
        self.medium_pattern = re.compile(
            r'\b(' + '|'.join(self.MEDIUM_URGENCY_KEYWORDS) + r')\b',
            re.IGNORECASE
        )
        self.low_pattern = re.compile(
            r'\b(' + '|'.join(self.LOW_URGENCY_KEYWORDS) + r')\b',
            re.IGNORECASE
        )
    
    def detect_urgency(self, text: str) -> float:
        """
        检测文本的紧迫性标量
        
        Args:
            text: 输入文本（对话、症状描述等）
        
        Returns:
            紧迫性标量 phi:
            - 1.5: 高危（检测到高危词汇）
            - 1.2: 中等紧迫
            - 1.0: 常规
            - 0.8: 低紧迫
        """
        if not text:
            return 1.0
            
        text_lower = str(text).lower()
        
        # 检测高危词汇
        if self.high_pattern.search(text_lower):
            return 1.5
        
        # 检测中等紧迫性
        if self.medium_pattern.search(text_lower):
            return 1.2
        
        # 检测低紧迫性
        if self.low_pattern.search(text_lower):
            return 0.8
        
        # 默认常规
        return 1.0
    
    def extract_urgency_from_event(self, event: Dict) -> float:
        """
        从事件中提取紧迫性
        
        Args:
            event: 医疗事件字典
        
        Returns:
            紧迫性标量
        """
        # 合并所有文本字段
        texts = []
        
        if event.get("summary"):
            texts.append(str(event["summary"]))
        
        if event.get("diagnosis"):
            texts.append(str(event["diagnosis"]))
        
        if event.get("symptoms"):
            texts.extend([str(s) for s in event.get("symptoms", [])])
        
        if event.get("raw_dialogue"):
            texts.extend([str(d) for d in event.get("raw_dialogue", [])])
        
        # 取最大紧迫性（最严重的情况）
        max_urgency = 1.0
        for text in texts:
            urgency = self.detect_urgency(str(text))
            max_urgency = max(max_urgency, urgency)
        
        return max_urgency

