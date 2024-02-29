from llmserve.backend.server.models import FTApp

def parse_task_name(ftapp: FTApp):
    task_purpose = (ftapp.ft_config.ft_task + "-") if ftapp.ft_config.ft_task else ""
    data_path = ftapp.ft_config.data_config.data_path
    data_name = ("-" + ftapp.ft_config.data_config.subset) if ftapp.ft_config.data_config.subset else ""

    return task_purpose + data_path + data_name
