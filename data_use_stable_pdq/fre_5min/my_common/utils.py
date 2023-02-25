import logging
def get_logger(log_file_name, log_level):
    # 创建logger对象
    logger = logging.getLogger(log_file_name)
    # 设置日志等级
    logger.setLevel(log_level)
    # 追加写入文件a ，设置utf-8编码防止中文写入乱码
    log = logging.FileHandler('log/' + log_file_name + '.log', 'a', encoding='utf-8')
    # 向文件输出的日志级别
    log.setLevel(log_level)
    # 向文件输出的日志信息格式
    formatter = logging.Formatter('%(asctime)s - %(filename)s - line:%(lineno)d - %(levelname)s - %(message)s')
    log.setFormatter(formatter)
    # 加载文件到logger对象中
    logger.addHandler(log)
    return logger