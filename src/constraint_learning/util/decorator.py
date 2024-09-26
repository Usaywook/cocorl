import time

def time_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 시작 시간 기록
        result = func(*args, **kwargs)  # 함수 실행
        end_time = time.time()  # 끝 시간 기록
        execution_time = end_time - start_time  # 실행 시간 계산
        print(f"Execution time: {execution_time:.6f} seconds")
        return result
    return wrapper()