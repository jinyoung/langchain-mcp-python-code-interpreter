"""
Short-term Memory System for MCP ReAct Client
Manages conversation history and context for interactive sessions
"""

import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class MemoryEntry:
    """메모리 엔트리 클래스"""
    timestamp: str
    user_input: str
    agent_response: str
    tools_used: List[str]
    session_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """딕셔너리에서 생성"""
        return cls(**data)


class ShortTermMemory:
    """단기 메모리 관리 클래스"""
    
    def __init__(self, session_id: str = None, max_entries: int = 50):
        """초기화"""
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.max_entries = max_entries
        self.memory: List[MemoryEntry] = []
        self.memory_file = Path("/Users/uengine/temp") / f"memory_{self.session_id}.json"
        
        # 기존 메모리 로드
        self.load_memory()
    
    def add_entry(self, user_input: str, agent_response: str, tools_used: List[str] = None) -> None:
        """새로운 메모리 엔트리 추가"""
        entry = MemoryEntry(
            timestamp=datetime.now().isoformat(),
            user_input=user_input,
            agent_response=agent_response,
            tools_used=tools_used or [],
            session_id=self.session_id
        )
        
        self.memory.append(entry)
        
        # 최대 엔트리 수 제한
        if len(self.memory) > self.max_entries:
            self.memory = self.memory[-self.max_entries:]
        
        # 자동 저장
        self.save_memory()
    
    def get_recent_entries(self, count: int = 5) -> List[MemoryEntry]:
        """최근 엔트리들 반환"""
        return self.memory[-count:] if count > 0 else self.memory
    
    def get_conversation_context(self, count: int = 5) -> str:
        """대화 컨텍스트 문자열 생성"""
        recent_entries = self.get_recent_entries(count)
        
        if not recent_entries:
            return "No previous conversation history."
        
        context_parts = []
        for entry in recent_entries:
            context_parts.append(f"User: {entry.user_input}")
            context_parts.append(f"Assistant: {entry.agent_response}")
            if entry.tools_used:
                context_parts.append(f"Tools used: {', '.join(entry.tools_used)}")
            context_parts.append("---")
        
        return "\n".join(context_parts)
    
    def search_memory(self, keyword: str) -> List[MemoryEntry]:
        """키워드로 메모리 검색"""
        results = []
        keyword_lower = keyword.lower()
        
        for entry in self.memory:
            if (keyword_lower in entry.user_input.lower() or 
                keyword_lower in entry.agent_response.lower()):
                results.append(entry)
        
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """메모리 요약 정보 반환"""
        if not self.memory:
            return {
                "total_entries": 0,
                "session_id": self.session_id,
                "first_interaction": None,
                "last_interaction": None,
                "most_used_tools": []
            }
        
        # 도구 사용 빈도 계산
        tool_usage = {}
        for entry in self.memory:
            for tool in entry.tools_used:
                tool_usage[tool] = tool_usage.get(tool, 0) + 1
        
        most_used_tools = sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_entries": len(self.memory),
            "session_id": self.session_id,
            "first_interaction": self.memory[0].timestamp,
            "last_interaction": self.memory[-1].timestamp,
            "most_used_tools": most_used_tools
        }
    
    def clear_memory(self) -> None:
        """메모리 초기화"""
        self.memory.clear()
        if self.memory_file.exists():
            self.memory_file.unlink()
    
    def save_memory(self) -> None:
        """메모리를 파일에 저장"""
        try:
            data = [entry.to_dict() for entry in self.memory]
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"메모리 저장 오류: {e}")
    
    def load_memory(self) -> None:
        """파일에서 메모리 로드"""
        try:
            if self.memory_file.exists():
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.memory = [MemoryEntry.from_dict(entry) for entry in data]
        except Exception as e:
            print(f"메모리 로드 오류: {e}")
            self.memory = []
    
    def export_memory(self, format: str = "json") -> str:
        """메모리를 다른 형식으로 내보내기"""
        if format.lower() == "json":
            return json.dumps([entry.to_dict() for entry in self.memory], 
                            ensure_ascii=False, indent=2)
        
        elif format.lower() == "text":
            lines = []
            lines.append(f"=== Session: {self.session_id} ===\n")
            
            for i, entry in enumerate(self.memory, 1):
                lines.append(f"[{i}] {entry.timestamp}")
                lines.append(f"User: {entry.user_input}")
                lines.append(f"Assistant: {entry.agent_response}")
                if entry.tools_used:
                    lines.append(f"Tools: {', '.join(entry.tools_used)}")
                lines.append("-" * 50)
            
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported format: {format}")


class MemoryManager:
    """메모리 관리자 클래스"""
    
    def __init__(self):
        self.current_memory: Optional[ShortTermMemory] = None
    
    def start_session(self, session_id: str = None) -> ShortTermMemory:
        """새 세션 시작"""
        self.current_memory = ShortTermMemory(session_id)
        return self.current_memory
    
    def get_current_memory(self) -> Optional[ShortTermMemory]:
        """현재 메모리 반환"""
        return self.current_memory
    
    def end_session(self) -> None:
        """현재 세션 종료"""
        if self.current_memory:
            self.current_memory.save_memory()
            self.current_memory = None
    
    def list_sessions(self) -> List[str]:
        """저장된 세션 목록 반환"""
        memory_dir = Path("/Users/uengine/temp")
        memory_files = memory_dir.glob("memory_session_*.json")
        return [f.stem.replace("memory_", "") for f in memory_files]

