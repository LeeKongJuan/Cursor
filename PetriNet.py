import xml.etree.ElementTree as ET
import pulp
from pyeda.inter import exprvars, expr2bdd, And, Or, bddvars

class PetriNet:
    def __init__(self):             #Khởi tạo các cấu trúc dữ liệu để lưu trữ thông tin Petri Net
        self.places = {}            #Lưu trữ địa điểm id:<name, tokens>
        self.transitions = {}       #Lưu trữ chuyển đổi id:<name>
        self.arcs = []              #Lưu trữ cung [(source, target_id, weight_id)]
        # adjacency lists (built after parsing PNML)
        self.input = {}
        self.output = {}
    """
    ==============================================================
    Phân tích tệp PNML và trích xuất thông tin Petri Net
    --------------------------------------------------------------
    File pnml có dạng:
    <pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
        <net id="net1" type="http://www.pnml.org/version-2009/grammar/ptnet">
            <page id="page1">
                <!-- Địa điểm-->
                    <place id="p1">                                         #id của địa điểm
                    <name><text>Place 1</text></name>                       #tên địa điểm
                    <initialMarking><text>1</text></initialMarking>         #số token ban đầu: 0 hoặc 1
                    </place>
                    ... Có thêm graphic để mô tả vị trí địa điểm trên giao diện (không cần phân tích) ...

                <!-- Chuyển đổi-->
                    <transition id="t1">                                    #id của chuyển đổi
                        <name><text>Transition 1</text></name>              #tên chuyển đổi
                    </transition>
                    ... Có thêm graphic để mô tả vị trí chuyển đổi trên giao diện (không cần phân tích) ...

                <!-- Cung-->
                    <arc id="a1" source="p1" target="t1">                   #id của cung, nguồn và đích
                    <inscription><text>2</text></inscription>               #trọng số của cung: số nguyên dương
                    </arc>
                    ... Có thêm graphic để mô tả vị trí cung trên giao diện (không cần phân tích) ...

            </page>
        </net>
    </pnml>
    --------------------------------------------------------------
    Note: validate arc với weight > 0 thì mới được thêm vào Petri Net
    Nếu nội dung nằm trong cùng 1 tag thì không cần pnml:namespace
    ==============================================================
    """
    def parse_pnml(self, file_name):
        tree = ET.parse(file_name)
        root = tree.getroot()
        ns = {'pnml': 'http://www.pnml.org/version-2009/grammar/pnml'}  #Định nghĩa namespace nếu cần thiết
        
        net = root.find('pnml:net', ns)
        page = net.find('pnml:page', ns)

        #Kiểm tra và trích xuất địa điểm, chuyển đổi, cung
        for place in page.findall('pnml:place', ns):
            place_id = place.get('id')
            place_name = place.findtext('pnml:name/pnml:text', default = place_id, namespaces = ns)
            initial_marking = int(place.findtext('pnml:initialMarking/pnml:text', default = 0, namespaces = ns))
            self.places[place_id] = {'name': place_name, 'tokens': initial_marking}

        for transition in page.findall('pnml:transition', ns):
            transition_id = transition.get('id')
            transition_name = transition.findtext('pnml:name/pnml:text', default=transition_id, namespaces = ns)
            self.transitions[transition_id] = {'name': transition_name}

        for arc in page.findall('pnml:arc', ns):
            source = arc.get('source')
            target = arc.get('target')
            weight = int(arc.findtext('pnml:inscription/pnmltext', default = 1, namespaces = ns))
            if (weight > 0): 
                self.arcs.append((source, target, weight))
        #Sau khi đọc xong thì xử lý việc tìm input và output cho từng transition
        self.build_input_output()

    def build_input_output(self):
        """Build input/output adjacency lists from `self.arcs` and
        `self.places`/`self.transitions`. This ensures the maps exist
        and are consistent even if parse_pnml runs after initialization.
        """
        # initialize dicts for every transition
        self.input = {t: [] for t in self.transitions}
        self.output = {t: [] for t in self.transitions}
        for (s, t, w) in self.arcs:
            if s in self.places and t in self.transitions:
                self.input[t].append(s)
            elif s in self.transitions and t in self.places:
                # output mapping keyed by transition ID; value is place ID
                self.output[s].append(t)

    """
    =============================================================
    Tính toán các trạng thái có thể đạt được bằng BFS (Explicit State Space)
    -------------------------------------------------------------
    Có 2 loại cung: cung từ địa điểm đến chuyển đổi (input arc) và cung từ chuyển đổi đến địa điểm (output arc)
    Một chuyển đổi có thể được kích hoạt nếu tất cả các địa điểm đầu vào của nó có đủ token theo trọng số của các cung tương ứng
    Khi một chuyển đổi được kích hoạt, nó sẽ tiêu thụ token từ các địa điểm đầu vào và tạo token tại các địa điểm đầu ra theo trọng số của các cung tương ứng
    -------------------------------------------------------------
    Dùng BFS để duyệt tất cả các trạng thái có thể đạt được
    Cần thêm 1 hàm phụ trợ: fire_transition để kích hoạt chuyển đổi và trả về đánh dấu mới
    =============================================================
    """
    def fire_transition(self, transition_id, marking):                  #Kích hoạt chuyển đổi và trả về đánh dấu mới
        new_marking = marking.copy()
        for (source, target, weight) in self.arcs:
            if(source in self.places and target == transition_id):      #Cung đầu vào: tiêu thụ token từ địa điểm nguồn
                if marking[source] < weight:                            #Nếu số token tại địa điểm nguồn nhỏ hơn trọng số cung, chuyển đổi không thể kích hoạt
                    return None
                new_marking[source] -= weight
            elif(source == transition_id and target in self.places):    #Cung đầu ra: tạo token tại địa điểm đích
                new_marking[target] += weight
        return new_marking

    def explicit_reachable_states(self):
        initial_marking = {place_id: self.places[place_id]['tokens'] for place_id in self.places}   #Đánh dấu ban đầu
        queue = [initial_marking]                                                                   #Đưa trạng thái đầu tiên của marking vào queue để xét
        visited = set()                                                                             #Lưu trữ các trạng thái đã xét để tránh bị lặp vô hạn
        visited.add(tuple(initial_marking.items()))
        reachable = []                                                                              #Lưu trữ tất cả các trạng thái có thể đạt được
        #Duyệt BFS
        while queue:
            current_marking = queue.pop(0)                             #Lấy trạng thái hiện tại từ queue
            reachable.append(current_marking)                          #Thêm trạng thái hiện tại vào danh sách các trạng thái có thể đạt được
            #Kiểm tra tất cả các chuyển đổi có thể kích hoạt từ trạng thái hiện tại
            for transition_id in self.transitions:
                new_marking = self.fire_transition(transition_id, current_marking)
                if new_marking is not None:
                    #Thêm trạng thái mới vào queue nếu chưa được xét
                    marking_tuple = tuple(new_marking.items())         #Chuyển đổi đánh dấu mới thành tuple để có thể thêm vào set
                    if marking_tuple not in visited:
                        visited.add(marking_tuple)
                        queue.append(new_marking)
        return reachable
    
    """
    ============================================================
    Tính toán các trạng thái có thể đạt được bằng Binary Decision Diagram (BDD)
    ------------------------------------------------------------
    Sử dụng thư viện pyeda để làm việc với BDD
    Mỗi địa điểm được biểu diễn bằng một biến Boolean trong BDD
    Sau đó, xây dựng biểu thức logic đại diện cho các chuyển đổi và trạng thái có thể đạt được
    ------------------------------------------------------------
    Các bước thực hiện:
    1. Chuẩn bị biến và marking ban đầu
    2. Xây dựng biểu thức logic cho từng chuyển đổi
    3. Kích hoạt chuyển đổi và cập nhật BDD. Lặp lại cho đến khi không còn trạng thái mới nào được thêm vào BDD
    ============================================================
    """
    def symbolic_reachable_states_1(self):
        # --- Bước 1: Chuẩn bị biến và marking ban đầu ---
        initial_marking = {place_id: self.places[place_id]['tokens'] for place_id in self.places}   #Đánh dấu ban đầu
        place_ids = list(self.places.keys())                                                        #Danh sách id của các địa điểm để ánh xạ với biến Boolean tương ứng
        p_cur = exprvars('p_cur', len(place_ids))                                                   #Tạo p biến Boolean đại diện cho từng điểm, p_cur = [p_cur0, p_cur1, p_cur2, ...]
        p_next = exprvars('p_next', len(place_ids))                                                 #Tương tự trên
        
        initial_expr = And(*[p_cur[i] if initial_marking[place_ids[i]] else ~p_cur[i] for i in range(len(self.places))])   #Biểu thức logic đại diện cho trạng thái ban đầu
        initial_bdd = expr2bdd(initial_expr)

        # --- Bước 2: Xây dựng biểu thức chuyển cho từng transition ---
        transition_each_exprs = []
        #Tạo biểu thức cho từng chuyển đổi
        for t_id in self.transitions:
            #Lấy dữ liệu của các biến vào và ra của transition đang xét
            inputs = set(self.input[t_id])
            outputs = set(self.output[t_id])
            # print(f"DEBUG: Transition {t_id} - inputs: {inputs}, outputs: {outputs}")
            #Dữ kiện để kích hoạt: toàn bộ input phải có token
            if inputs:      #Nếu input rỗng thì trả về True, vì nó luôn được kích hoạt
                enabled_expr = And(*[p_cur[place_ids.index(p)] for p in inputs])
                # print(f"DEBUG: enabled_expr: {enabled_expr}")
            else:
                enabled_expr = And()
                # print(f"DEBUG: enabled_expr: True (no inputs)")

            #Đánh dấu Marking sau khi bắn: input sẽ mất token, output nhận token
            next_state_exprs = []
            i = 0
            for p_id in place_ids:
                if p_id in inputs and p_id in outputs:
                    #Self-loop: giữ nguyên token (mất rồi được lại)
                    next_state_exprs.append(~(p_next[i] ^ p_cur[i]))
                elif p_id in inputs:  
                    #Nếu là input
                    next_state_exprs.append(~p_next[i])
                elif p_id in outputs: 
                    #Nếu là output
                    next_state_exprs.append(p_next[i])
                else: 
                    #Nếu không liên quan thì giữ nguyên
                    next_state_exprs.append(~(p_next[i] ^ p_cur[i]))
                i += 1
            
            #Kết hợp lại các công thức ở trên thành dạng T = (Điều kiện kích hoạt) AND (Marking sau khi kích hoạt)
            t_expr = And(enabled_expr, *next_state_exprs)
            # print(f"DEBUG: t_expr: {t_expr}")
            transition_each_exprs.append(t_expr)
        
        #Kết hợp lại toàn bộ biểu thức transition với Or
        # print(f"Debug: transition full: {transition_exprs}")
        transition_bdd = expr2bdd(Or(*transition_each_exprs))
        
        # --- Bước 3: Kích hoạt chuyển đổi và cập nhật BDD. Lặp lại cho đến khi không còn trạng thái mới nào được thêm vào BDD ---
        reachable = initial_bdd
        cur_states = initial_bdd
        
        # Chuyển sang BDD
        p_cur_bdd = bddvars('p_cur', len(place_ids))
        p_next_bdd = bddvars('p_next', len(place_ids))
        rename_dict = {p_next_bdd[i]: p_cur_bdd[i] for i in range(len(place_ids))}

        iter = 0
        while True:
            # Sau khi tính transition_bdd và post
            post = (cur_states & transition_bdd)
            # print(f"DEBUG: post: {bdd2expr(post)}")
            # Smoothing loại bỏ các biến p_cur
            post = post.smoothing(p_cur_bdd)
            # print(f"DEBUG: post: {bdd2expr(post)}")
            # Compose để p_next -> p_cur
            post = post.compose(rename_dict)
            # print(f"DEBUG: post: {bdd2expr(post)}")
            # Hợp vào tập reachable
            updated = reachable | post

            # print(f"DEBUG: post: {bdd2expr(updated)}")
            if updated.equivalent(reachable):
                break

            cur_states = updated^reachable
            # print(f"DEBUG: {bdd2expr(cur_states)}")
            reachable = updated

            iter += 1
        # print(f"DEBUG: {bdd2expr(reachable)}")
        return reachable

    def symbolic_reachable_states_2(self):
        # --- Bước 1: Chuẩn bị biến và marking ban đầu ---
        initial_marking = {place_id: self.places[place_id]['tokens'] for place_id in self.places}   #Đánh dấu ban đầu
        place_ids = list(self.places.keys())                                                        #Danh sách id của các địa điểm để ánh xạ với biến Boolean tương ứng
        p_cur = exprvars('p_cur', len(place_ids))                                                   #Tạo p biến Boolean đại diện cho từng điểm, p_cur = [p_cur0, p_cur1, p_cur2, ...]
        p_next = exprvars('p_next', len(place_ids))                                                 #Tương tự trên
        
        initial_expr = And(*[p_cur[i] if initial_marking[place_ids[i]] else ~p_cur[i] for i in range(len(self.places))])   #Biểu thức logic đại diện cho trạng thái ban đầu
        initial_bdd = expr2bdd(initial_expr)

        # --- Bước 2: Xây dựng biểu thức chuyển cho từng transition ---
        transition_bdd = []
        #Tạo biểu thức cho từng chuyển đổi
        for t_id in self.transitions:
            #Lấy dữ liệu của các biến vào và ra của transition đang xét
            inputs = set(self.input[t_id])
            outputs = set(self.output[t_id])
            # print(f"DEBUG: Transition {t_id} - inputs: {inputs}, outputs: {outputs}")
            #Dữ kiện để kích hoạt: toàn bộ input phải có token
            if inputs:      #Nếu input rỗng thì trả về True, vì nó luôn được kích hoạt
                enabled_expr = And(*[p_cur[place_ids.index(p)] for p in inputs])
                # print(f"DEBUG: enabled_expr: {enabled_expr}")
            else:
                enabled_expr = And()
                # print(f"DEBUG: enabled_expr: True (no inputs)")

            #Đánh dấu Marking sau khi bắn: input sẽ mất token, output nhận token
            next_state_exprs = []
            i = 0
            for p_id in place_ids:
                if p_id in inputs and p_id in outputs:
                    #Self-loop: giữ nguyên token (mất rồi được lại)
                    next_state_exprs.append(~(p_next[i] ^ p_cur[i]))
                elif p_id in inputs:  
                    #Nếu là input
                    next_state_exprs.append(~p_next[i])
                elif p_id in outputs: 
                    #Nếu là output
                    next_state_exprs.append(p_next[i])
                else: 
                    #Nếu không liên quan thì giữ nguyên
                    next_state_exprs.append(~(p_next[i] ^ p_cur[i]))
                i += 1
            
            #Kết hợp lại các công thức ở trên thành dạng T = (Điều kiện kích hoạt) AND (Marking sau khi kích hoạt)
            t_expr = And(enabled_expr, *next_state_exprs)
            transition_bdd.append(expr2bdd(t_expr))
        
        # --- Bước 3: Kích hoạt chuyển đổi và cập nhật BDD. Lặp lại cho đến khi không còn trạng thái mới nào được thêm vào BDD ---
        reachable = initial_bdd
        cur_states = initial_bdd
        
        # Chuyển sang BDD
        p_cur_bdd = bddvars('p_cur', len(place_ids))
        p_next_bdd = bddvars('p_next', len(place_ids))
        rename_dict = {p_next_bdd[i]: p_cur_bdd[i] for i in range(len(place_ids))}

        iter = 0
        while True:
            post_total = 0  # false BDD
            for t_bdd in transition_bdd:    # Lần lượt cập nhật giá trị hiện tại với các transition, với các hàm sử dụng tương tự trên
                post = (cur_states & t_bdd)
                post = post.smoothing(p_cur_bdd)
                post = post.compose(rename_dict)
                post_total |= post

            updated = reachable | post_total

            # print(f"DEBUG: post: {bdd2expr(updated)}")
            if updated.equivalent(reachable):
                break

            cur_states = updated^reachable
            # print(f"DEBUG: {bdd2expr(cur_states)}")
            reachable = updated

            iter += 1
        # print(f"DEBUG: {bdd2expr(reachable)}")
        return reachable

    def symbolic_reachable_states_3(self):
        # --- Bước 1: Chuẩn bị biến và marking ban đầu ---
        initial_marking = {place_id: self.places[place_id]['tokens'] for place_id in self.places}
        place_ids = list(self.places.keys())

        p_cur = bddvars('p_cur', len(place_ids))
        p_next = bddvars('p_next', len(place_ids))

        # BDD cho marking ban đầu
        initial_bdd = 1
        for i in range(len(place_ids)):
            if initial_marking[place_ids[i]]:
                initial_bdd &= p_cur[i]
            else:
                initial_bdd &= ~p_cur[i]

        # --- Bước 2: Xây dựng BDD transition ---
        transition_bdd = []

        for t_id in self.transitions:
            inputs = set(self.input[t_id])
            outputs = set(self.output[t_id])

            # enabled condition
            enabled_bdd = 1
            for p in inputs:
                enabled_bdd &= p_cur[place_ids.index(p)]

            # next-state constraints
            next_state_bdd = 1
            for i, p_id in enumerate(place_ids):
                if p_id in inputs and p_id in outputs:
                    next_state_bdd &= ~(p_next[i] ^ p_cur[i])
                elif p_id in inputs:
                    next_state_bdd &= ~p_next[i]
                elif p_id in outputs:
                    next_state_bdd &= p_next[i]
                else:
                    next_state_bdd &= ~(p_next[i] ^ p_cur[i])

            # tổng hợp thành transition BDD
            t_bdd = enabled_bdd & next_state_bdd
            transition_bdd.append(t_bdd)

        # --- Bước 3: Duyệt reachable states ---
        reachable = initial_bdd
        cur_states = initial_bdd

        # rename p_next -> p_cur
        rename_dict = {p_next[i]: p_cur[i] for i in range(len(place_ids))}

        while True:
            post_total = 0

            for t_bdd in transition_bdd:
                post = cur_states & t_bdd
                post = post.smoothing(p_cur)
                post = post.compose(rename_dict)
                post_total |= post

            updated = reachable | post_total

            if updated.equivalent(reachable):
                break

            cur_states = updated ^ reachable
            reachable = updated

        return reachable
    
    """
    ============================================================
    ILP for finding deadlock
    ============================================================
    """
    def find_reachable_deadlock_ilp(self):
        #Tính toán tập trạng thái khả đạt R (kế thừa Explicit ở trên)
        reachable_states = self.explicit_reachable_states()
        
        if not reachable_states:
            print("Không có trạng thái khả đạt nào.")
            return None
        #Lấy danh sách ID của các place trong mạng. Danh sách này được dùng để định nghĩa ILP
        place_ids = list(self.places.keys())
        k = len(reachable_states) #Lấy số lượng trạng thái khả đạt
        
        #Khởi tạo bài toán ILP
        prob = pulp.LpProblem("Reachable_Deadlock_Search", pulp.LpMinimize)
        
        #Định nghĩa biến quyết định (M_p) và biến tuyển chọn (Y_j)
        M = pulp.LpVariable.dicts("M", place_ids, 0, 1, pulp.LpBinary)    
        Y = pulp.LpVariable.dicts("Y", range(k), 0, 1, pulp.LpBinary)

        #Thiết lập hàm mục tiêu là tối thiểu hóa số token trong đánh dấu M
        prob += pulp.lpSum(M[p] for p in place_ids), "Total_Tokens"

        #Ràng buộc Khả Đạt
        
        #Chỉ chọn MỘT trạng thái từ R
        prob += pulp.lpSum(Y[j] for j in range(k)) == 1, "Select_One_Reachable_State"
        for p_id in place_ids:
            M_p_values_sum = pulp.lpSum(
                reachable_states[j][p_id] * Y[j] for j in range(k)
            )
            prob += M[p_id] == M_p_values_sum, f"Link_Marking_Place_{p_id}"
        #Ràng buộc Deadlock 
        for t_id in self.transitions:
            input_places = []
            for (source, target, weight) in self.arcs:
                if target == t_id and source in self.places:
                    input_places.append(source)
            
            num_inputs = len(input_places)

            if num_inputs > 0:
                #Ràng buộc VÔ HIỆU HÓA: sum(M[p]) <= num_inputs - 1
                prob += pulp.lpSum(M[p] for p in input_places) <= num_inputs - 1, f"Disable_Transition_{t_id}"
            else:
                #Nếu có transition không input, không thể có deadlock.
                print(f"Lưu ý: Transition {t_id} không có place đầu vào. Mạng này không thể có deadlock.")
                return None 

        #Giải bài toán ILP
        try:
            solver = pulp.getSolver("PULP_CBC_CMD", msg=0)
            prob.solve(solver)
            
            #Kiểm tra và trả về kết quả
            if pulp.LpStatus[prob.status] == "Optimal" or pulp.LpStatus[prob.status] == "Feasible":
                deadlock_marking = {p_id: int(pulp.value(M[p_id])) for p_id in place_ids}
                return deadlock_marking
            else:
                return None
        except Exception as e:
            print(f"Lỗi khi giải ILP: {e}")
            return None

    """
    ============================================================
    OPTIMIZATION
    ============================================================
    """ 
    def optimize_reachable_marking_ilp(self, c_weights):
        reachable_states = self.explicit_reachable_states() 
    
        if not reachable_states:
            print("Không có trạng thái khả đạt nào.")
            return None, None
        
        place_ids = list(self.places.keys())
        k = len(reachable_states) 
    
        prob = pulp.LpProblem("Maximize_Reachable_Marking", pulp.LpMaximize)
    
    
        M = pulp.LpVariable.dicts("M", place_ids, 0, 1, pulp.LpBinary) 
        Y = pulp.LpVariable.dicts("Y", range(k), 0, 1, pulp.LpBinary)

        objective_sum = pulp.lpSum(c_weights.get(p, 0) * M[p] for p in place_ids)
        prob += objective_sum, "Weighted_Total_Tokens"
        prob += pulp.lpSum(Y[j] for j in range(k)) == 1, "Select_One_Reachable_State"
    
        for p_id in place_ids:
            M_p_values_sum = pulp.lpSum(
                reachable_states[j].get(p_id, 0) * Y[j] for j in range(k)
            )
            prob += M[p_id] == M_p_values_sum, f"Link_Marking_Place_{p_id}"
        
        try:
            solver = pulp.getSolver("PULP_CBC_CMD", msg=0)
            prob.solve(solver)
            
            if pulp.LpStatus[prob.status] == "Optimal" or pulp.LpStatus[prob.status] == "Feasible":
                optimal_marking = {p_id: int(pulp.value(M[p_id])) for p_id in place_ids}
                optimal_value = pulp.value(prob.objective)
                print(f"Giá trị tối ưu: {optimal_value}")
                return optimal_marking, optimal_value
            else:
                print("Không tìm thấy Đánh dấu khả đạt tối ưu.")
                return None, None
        except Exception as e:
            print(f"Lỗi khi giải ILP: {e}")
            return None, None