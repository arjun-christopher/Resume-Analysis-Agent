# app/utils.py - Enhanced utility functions with better error handling
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_session_paths(force: bool = False) -> Dict[str, str]:
    """Initialize session-specific paths with better organization"""
    base = Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / "rag_resume"
    base.mkdir(parents=True, exist_ok=True)

    # Generate unique session ID
    session_id = uuid.uuid4().hex[:12]
    session_root = base / session_id
    session_root.mkdir(parents=True, exist_ok=True)

    paths = {
        "session_root": str(session_root),
        "uploads": str(session_root / "uploads"),
        "vector": str(session_root / "vector"),
        "temp": str(session_root / "temp"),
        "logs": str(session_root / "logs"),
        "cache": str(session_root / "cache")
    }
    
    # Create all directories
    for key, path_str in paths.items():
        if key != "session_root":
            Path(path_str).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Initialized session paths with ID: {session_id}")
    return paths

def max_upload_guard(total: int, max_allowed: int = 50) -> bool:
    """Guard against too many uploads with better error handling"""
    if total > max_allowed:
        logger.warning(f"Upload limit exceeded: {total} > {max_allowed}")
        return False
    return True

def human_size(n: int) -> str:
    """Convert bytes to human-readable format with better precision"""
    if not isinstance(n, (int, float)) or n < 0:
        return "0B"
    
    size = float(n)
    units = ["B", "KB", "MB", "GB", "TB", "PB", "EB"]
    
    unit_index = 0
    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    
    if unit_index == 0:
        return f"{int(size)}B"
    elif size < 10:
        return f"{size:.2f}{units[unit_index]}"
    elif size < 100:
        return f"{size:.1f}{units[unit_index]}"
    else:
        return f"{int(size)}{units[unit_index]}"

def list_supported_files(uploads_dir: Union[str, Path]) -> List[Path]:
    """List all supported files in directory with error handling"""
    try:
        path = Path(uploads_dir)
        if not path.exists():
            logger.warning(f"Directory does not exist: {uploads_dir}")
            return []
        
        supported_extensions = {".pdf", ".docx", ".png", ".jpg", ".jpeg"}
        files = []
        
        for file_path in path.iterdir():
            if (file_path.is_file() and 
                file_path.suffix.lower() in supported_extensions and
                not file_path.name.startswith('.')):  # Skip hidden files
                files.append(file_path)
        
        return sorted(files, key=lambda x: x.name.lower())
    
    except Exception as e:
        logger.error(f"Error listing files in {uploads_dir}: {e}")
        return []

def safe_listdir(directory: Union[str, Path]) -> List[Path]:
    """Safely list directory contents with error handling"""
    try:
        path = Path(directory)
        if not path.exists():
            return []
        
        return [p for p in path.iterdir() if p.is_file()]
    
    except Exception as e:
        logger.error(f"Error accessing directory {directory}: {e}")
        return []

def clear_dir(path: Union[str, Path], preserve_structure: bool = False) -> bool:
    """Clear directory contents with better error handling"""
    try:
        path_obj = Path(path)
        if not path_obj.exists():
            logger.info(f"Directory does not exist: {path}")
            return True
        
        if preserve_structure:
            # Only remove files, keep subdirectories
            for item in path_obj.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
        else:
            # Remove everything
            shutil.rmtree(path_obj, ignore_errors=True)
            path_obj.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Successfully cleared directory: {path}")
        return True
    
    except Exception as e:
        logger.error(f"Error clearing directory {path}: {e}")
        return False

def validate_file_size(file_path: Union[str, Path], max_size_mb: int = 100) -> bool:
    """Validate file size is within limits"""
    try:
        path = Path(file_path)
        if not path.exists():
            return False
        
        size_mb = path.stat().st_size / (1024 * 1024)
        return size_mb <= max_size_mb
    
    except Exception as e:
        logger.error(f"Error validating file size for {file_path}: {e}")
        return False

def validate_file_type(file_path: Union[str, Path], allowed_extensions: set = None) -> bool:
    """Validate file type against allowed extensions"""
    if allowed_extensions is None:
        allowed_extensions = {".pdf", ".docx", ".png", ".jpg", ".jpeg", ".zip"}
    
    try:
        path = Path(file_path)
        return path.suffix.lower() in allowed_extensions
    
    except Exception as e:
        logger.error(f"Error validating file type for {file_path}: {e}")
        return False

def create_backup(source_dir: Union[str, Path], backup_dir: Union[str, Path] = None) -> Optional[Path]:
    """Create backup of directory"""
    try:
        source = Path(source_dir)
        if not source.exists():
            logger.warning(f"Source directory does not exist: {source_dir}")
            return None
        
        if backup_dir is None:
            backup_dir = source.parent / f"{source.name}_backup_{uuid.uuid4().hex[:8]}"
        
        backup_path = Path(backup_dir)
        shutil.copytree(source, backup_path)
        
        logger.info(f"Created backup: {source} -> {backup_path}")
        return backup_path
    
    except Exception as e:
        logger.error(f"Error creating backup of {source_dir}: {e}")
        return None

def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Get comprehensive file information"""
    try:
        path = Path(file_path)
        if not path.exists():
            return {"error": "File does not exist"}
        
        stat_info = path.stat()
        
        return {
            "name": path.name,
            "stem": path.stem,
            "suffix": path.suffix.lower(),
            "size": stat_info.st_size,
            "size_human": human_size(stat_info.st_size),
            "created": stat_info.st_ctime,
            "modified": stat_info.st_mtime,
            "is_file": path.is_file(),
            "is_dir": path.is_dir(),
            "absolute_path": str(path.absolute())
        }
    
    except Exception as e:
        logger.error(f"Error getting file info for {file_path}: {e}")
        return {"error": str(e)}

def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """Sanitize filename to be safe for file system"""
    import re
    
    # Remove or replace dangerous characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove control characters
    sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', sanitized)
    
    # Trim whitespace and dots
    sanitized = sanitized.strip(' .')
    
    # Ensure it's not empty
    if not sanitized:
        sanitized = "unnamed_file"
    
    # Truncate if too long
    if len(sanitized) > max_length:
        name, ext = os.path.splitext(sanitized)
        max_name_length = max_length - len(ext)
        sanitized = name[:max_name_length] + ext
    
    return sanitized

def count_files_by_type(directory: Union[str, Path]) -> Dict[str, int]:
    """Count files by their extensions"""
    try:
        path = Path(directory)
        if not path.exists():
            return {}
        
        file_counts = {}
        for file_path in path.iterdir():
            if file_path.is_file():
                ext = file_path.suffix.lower()
                file_counts[ext] = file_counts.get(ext, 0) + 1
        
        return file_counts
    
    except Exception as e:
        logger.error(f"Error counting files in {directory}: {e}")
        return {}

def calculate_directory_size(directory: Union[str, Path]) -> int:
    """Calculate total size of directory"""
    try:
        path = Path(directory)
        if not path.exists():
            return 0
        
        total_size = 0
        for file_path in path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return total_size
    
    except Exception as e:
        logger.error(f"Error calculating directory size for {directory}: {e}")
        return 0

def is_safe_path(path: Union[str, Path], base_path: Union[str, Path]) -> bool:
    """Check if path is safe (within base directory, no path traversal)"""
    try:
        abs_path = Path(path).resolve()
        abs_base = Path(base_path).resolve()
        
        # Check if the path is within the base directory
        return abs_path.is_relative_to(abs_base)
    
    except Exception as e:
        logger.error(f"Error checking path safety: {e}")
        return False

def cleanup_temp_files(temp_dir: Union[str, Path], max_age_hours: int = 24) -> int:
    """Clean up temporary files older than specified age"""
    try:
        import time
        
        path = Path(temp_dir)
        if not path.exists():
            return 0
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        cleaned_count = 0
        
        for file_path in path.rglob('*'):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                    except Exception as e:
                        logger.warning(f"Could not remove old temp file {file_path}: {e}")
        
        logger.info(f"Cleaned up {cleaned_count} temporary files")
        return cleaned_count
    
    except Exception as e:
        logger.error(f"Error cleaning up temp files: {e}")
        return 0

class FileManager:
    """Enhanced file management class with comprehensive operations"""
    
    def __init__(self, base_dir: Union[str, Path], max_files: int = 50):
        self.base_dir = Path(base_dir)
        self.max_files = max_files
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def add_files(self, files: List[Union[str, Path]], move: bool = False) -> List[Path]:
        """Add files to managed directory"""
        added_files = []
        
        for file_path in files:
            source = Path(file_path)
            if not source.exists():
                logger.warning(f"Source file does not exist: {file_path}")
                continue
            
            # Generate safe destination name
            dest_name = sanitize_filename(source.name)
            dest_path = self.base_dir / dest_name
            
            # Handle name conflicts
            counter = 1
            original_dest = dest_path
            while dest_path.exists():
                name, ext = original_dest.stem, original_dest.suffix
                dest_path = self.base_dir / f"{name}_{counter}{ext}"
                counter += 1
            
            try:
                if move:
                    shutil.move(str(source), str(dest_path))
                else:
                    shutil.copy2(str(source), str(dest_path))
                
                added_files.append(dest_path)
                logger.info(f"Added file: {dest_path}")
            
            except Exception as e:
                logger.error(f"Error adding file {source}: {e}")
        
        return added_files
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of managed files"""
        files = list_supported_files(self.base_dir)
        total_size = sum(f.stat().st_size for f in files)
        file_types = count_files_by_type(self.base_dir)
        
        return {
            "total_files": len(files),
            "total_size": total_size,
            "total_size_human": human_size(total_size),
            "file_types": file_types,
            "files": [get_file_info(f) for f in files[:20]]  # Limit for performance
        }
    
    def cleanup(self) -> bool:
        """Clean up all managed files"""
        return clear_dir(self.base_dir, preserve_structure=True)