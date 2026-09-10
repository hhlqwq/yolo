"""流水线异常定义."""


class PipelineError(RuntimeError):
    """表示可向用户说明并安全终止的流水线错误."""


class ExcelPendingError(PipelineError):
    """模型结果已安全保存,但中心 Excel 暂时无法写入."""
