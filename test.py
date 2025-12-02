from PetriNet import PetriNet
from pyeda.inter import bdd2expr, exprvars
import time, psutil, os
import threading

def measure_function(func, *args, **kwargs):
    """
    Đo lường hiệu suất thực thi của một hàm: thời gian chạy và mức sử dụng bộ nhớ cao nhất.
    Args:
        func: Hàm cần đo lường hiệu suất
        *args: Các đối số vị trí truyền vào hàm
        **kwargs: Các đối số từ khóa truyền vào hàm
    Returns:
        tuple: (kết quả hàm, thời gian chạy (giây), bộ nhớ cao nhất (MB))
    """
    # Lấy thông tin process hiện tại
    process = psutil.Process(os.getpid())
    peak_mem = 0  # Theo dõi bộ nhớ cao nhất
    running = True  # Cờ điều khiển vòng lặp monitor

    def monitor():
        """
        Hàm chạy trong thread riêng để liên tục theo dõi mức sử dụng bộ nhớ.
        """
        nonlocal peak_mem  # Truy cập biến peak_mem từ outer scope
        while running:  # Chạy cho đến khi running = False
            mem = process.memory_info().rss  # Lấy bộ nhớ thực (resident set size)
            peak_mem = max(peak_mem, mem)  # Cập nhật giá trị cao nhất
            time.sleep(0.001)  # Ngủ 1ms để giảm tải CPU

    # Tạo và chạy thread monitor
    t = threading.Thread(target=monitor)
    t.start()
    # Đo thời gian thực thi hàm
    start_time = time.time()
    result = func(*args, **kwargs)  # Gọi hàm cần đo lường
    end_time = time.time()
    # Dừng thread monitor
    running = False  # Báo hiệu dừng vòng lặp
    t.join()  # Đợi thread kết thúc
    # Trả về kết quả: kết quả hàm, thời gian chạy, bộ nhớ cao nhất (MB)
    return result, end_time - start_time, peak_mem / 1_000_000

def print_time_memo(time, mem):
    print("Performance:")
    print(f"Thời gian: {time:.6f} s")
    print(f"Bộ nhớ: {mem:.3f} MB")

# --- Task 1: Parse file ---
def parse_result(pn):
    print("Địa điểm và token ban đầu:")
    for place_id, place_info in pn.places.items():
        print(f"- {place_id} ({place_info['name']}): {place_info['tokens']} token(s)")     #In place dưới dạng: ID (tên): số token(s)

    print("\nChuyển đổi:")
    for transition_id, transition_info in pn.transitions.items():
        print(f"- {transition_id} ({transition_info['name']})")                            #In transition dưới dạng: ID (tên)

    print("\nCung:")
    for arc in pn.arcs:
        print(f"- {arc[0]} -> {arc[1]} (weight: {arc[2]})")                                #In arc dưới dạng: nguồn -> đích (trọng số)

# --- Task 2: Explicit ---
def explicit(ex_reachable_states):
    print("\n=== KẾT QUẢ EXPLICIT ===")
    print("Trạng thái có thể đạt được:")
    for state in ex_reachable_states:
        print(state)

# --- Task 3: Symbolic ---
def convert_BDD_to_dict(sym_reachable_states) -> list:
    place_ids = list(pn.places.keys())                                                        #Danh sách id của các địa điểm để ánh xạ với biến Boolean tương ứng
    all_models = list(sym_reachable_states.satisfy_all())
    all_models.reverse()  # Đảo ngược thứ tự
    bdd_states = []
    for model in all_models:
        state_dict = {}
        for i, place_id in enumerate(place_ids):
            var_name = f"p_cur[{i}]"
            #Tìm key có tên trùng var_name
            matched_key = None
            for k in model.keys():
                #So sánh tên chuỗi của biến trong model với "p_cur[i]"
                if str(k) == var_name:
                    matched_key = k
                    break
            value = model[matched_key]

            state_dict[place_id] = 1 if matched_key and value else 0
        bdd_states.append(state_dict)
    return bdd_states

def symbolic(sym_reachable_states):
    print("\n=== KẾT QUẢ SYMBOLIC ===")
    print("Trạng thái có thể đạt được:")
    state_dict = convert_BDD_to_dict(sym_reachable_states)
    for state in state_dict:
        print(state)

# So sánh hiệu suất hiệu năng và kết quả
def compare_helper(explicit_states, bdd_states):
    """
    So sánh hai tập trạng thái (dưới dạng list[dict]) bất kể thứ tự.
    In ra các trạng thái chỉ có ở một bên nếu khác nhau.
    """
    # Chuyển dict → tuple để so sánh dễ hơn
    explicit_set = {tuple(sorted(state.items())) for state in explicit_states}
    bdd_set = {tuple(sorted(state.items())) for state in bdd_states}

    only_in_explicit = explicit_set - bdd_set
    only_in_bdd = bdd_set - explicit_set

    if not only_in_explicit and not only_in_bdd:
        print("✅ Hai phương pháp cho kết quả giống nhau (cùng tập trạng thái).")
    else:
        print("⚠️ Kết quả khác nhau!")
        if only_in_explicit:
            print("\nChỉ có trong explicit:")
            for s in only_in_explicit:
                print(dict(s))
        if only_in_bdd:
            print("\nChỉ có trong symbolic:")
            for s in only_in_bdd:
                print(dict(s))

# --- So sánh ---
def compare_result(pn, ex_reachable_states, ex_time, ex_mem, sym_reachable_states, sym_time1, sym_mem1, sym_time2, sym_mem2):
    bdd_states = convert_BDD_to_dict(sym_reachable_states)
    print("\n=== BÁO CÁO SO SÁNH HIỆU NĂNG ===")
    print(f"Số Places: {len(pn.places)}, Số Transitions: {len(pn.transitions)}, Số Arcs: {len(pn.arcs)}")
    print(f"Tổng số trạng thái (explicit): {len(ex_reachable_states)}")
    print(f"Tổng số trạng thái (symbolic): {len(bdd_states)}")
    compare_helper(ex_reachable_states, bdd_states)

    print(f"\n--- Thời gian ---")
    print(f"Explicit: {ex_time:.6f} s")
    print(f"Symbolic 1: {sym_time1:.6f} s")
    print(f"Symbolic 2: {sym_time2:.6f} s")
    print(f"\n--- Bộ nhớ ---")
    print(f"Explicit: {ex_mem:.3f} MB")
    print(f"Symbolic 1: {sym_mem1:.3f} MB")
    print(f"Symbolic 2: {sym_mem2:.3f} MB")

# --- Task 4: Deadlock Detection ---
def deadlock_detection(pn):
    print("\n=== KẾT QUẢ ILP DEADLOCK ===")
    deadlock_marking = pn.find_reachable_deadlock_ilp()

    if deadlock_marking is not None:
        # Định dạng kết quả để dễ đọc
        formatted_marking = {
            pn.places[p_id]['name']: token 
            for p_id, token in deadlock_marking.items()
        }
        print("Trạng thái Deadlock được tìm thấy:")
        print(deadlock_marking)
        print("Định dạng dễ đọc:")
        print(formatted_marking)
    else:
        print("Không tìm thấy trạng thái deadlock")

    ilp_deadlock_result, ilp_time, ilp_mem = measure_function(pn.find_reachable_deadlock_ilp)
    deadlock_marking = ilp_deadlock_result
    # --- In kết quả đo lường ILP ---
    print(f"\n--- Hiệu năng ILP Deadlock ---")
    print(f"Thời gian: {ilp_time:.6f} s")
    print(f"Bộ nhớ: {ilp_mem:.3f} MB")

# --- Task 5: Optimization Over Reachable Markings
def optimization(pn):
    c_weights = {}

    # Nhập vector c
    place_ids = list(pn.places.keys())

    for p_id in place_ids:
        place_name = pn.places[p_id].get('name', p_id) 
        
        while True:
            try:
                # Yêu cầu người dùng nhập giá trị cho Place này
                prompt = f"Nhập trọng số c cho Place ID '{p_id}' ({place_name}): "
                
                # Đọc input 
                weight = float(input(prompt))
                
                c_weights[p_id] = weight
                break 
                
            except ValueError:
                print("Lỗi: Vui lòng nhập một giá trị số hợp lệ (ví dụ: 10, -5, 3.5).")

    ilp_opt_result, ilp_opt_time, ilp_opt_mem = measure_function(
        pn.optimize_reachable_marking_ilp, c_weights
    )

    optimal_marking, optimal_value = ilp_opt_result

    print("Vector Trọng số c sử dụng:")
    print(c_weights)

    if optimal_marking is not None:
        # Định dạng kết quả để dễ đọc
        formatted_marking = {
            pn.places[p_id]['name']: token 
            for p_id, token in optimal_marking.items()
        }
        print("\nTrạng thái Tối ưu được tìm thấy:")
        print(optimal_marking)
        print("Định dạng dễ đọc:")
        print(formatted_marking)
        print(f"\nGiá trị Tối ưu: {optimal_value}")
    else:
        print("Không tìm thấy trạng thái khả đạt tối ưu (có thể tập khả đạt rỗng).")

    # --- In kết quả đo lường ILP Optimization ---
    print(f"\n--- Hiệu năng ILP Optimization ---")
    print(f"Thời gian: {ilp_opt_time:.6f} s")
    print(f"Bộ nhớ: {ilp_opt_mem:.3f} MB")



"""
Main, gọi các function ở trên để check
"""
pn = PetriNet()
# --- Gọi hàm để test ---
pn.parse_pnml("Petrinet_4.pnml")
# parse_result(pn)
# # --- Chạy và đo explicit và symbolic ---
# # --- Explicit ---
# ex_reachable_states, ex_time, ex_mem = measure_function(pn.explicit_reachable_states)
# explicit(ex_reachable_states)
# print_time_memo(ex_time, ex_mem)
# # --- Symbolic ---
# sym_reachable_states_1, sym_time_1, sym_mem_1 = measure_function(pn.symbolic_reachable_states_1)
# symbolic(sym_reachable_states_1)
# print("\n=== Cách 1 === ")
# print_time_memo(sym_time_1, sym_mem_1)
sym_reachable_states_2, sym_time_2, sym_mem_2 = measure_function(pn.symbolic_reachable_states_2)
print("\n=== Cách 2 ===")
print_time_memo(sym_time_2, sym_mem_2)
# if sym_reachable_states_1 == sym_reachable_states_2:
#     print("✅ Hai cách của symbolic cho kết quả giống nhau.")
# else:
#     print("⚠️ Kết quả khác nhau!")
# # --- So sánh ---
# compare_result(pn, ex_reachable_states, ex_time, ex_mem, sym_reachable_states_1, sym_time_1, sym_mem_1, sym_time_2, sym_mem_2)
# # --- Deadlock ---
# deadlock_detection(pn)
# # --- Optimization ---
# optimization(pn)
