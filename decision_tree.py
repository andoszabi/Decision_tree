import pandas as pd
import itertools as it

class Node():
    def __init__(self, index, parent = None, last_branching = None):
        self.index = index
        self.parent = parent
        self.last_branching = last_branching
        self.column_name = None
        self.column_value = None
        self.child1 = None
        self.child2 = None
        self.average = None
        self.leaf = True

    def add_data(self, column_name, column_value):
        self.column_name = column_name
        self.column_value = column_value

    def add_children(self, child1, child2):
        self.child1 = child1
        self.child2 = child2
        self.leaf = False

    def add_average(self, average):
        self.average = average

class DecisionTree():
    def __init__(self, data_frame, target_col):
        self.starting_node = Node(0)
        self.nodes = [self.starting_node]
        self.current_node = self.starting_node
        self.data_frame = data_frame
        self.done_building_tree = False
        self.target_col = target_col
        self.__build_tree()

    def __add_nodes(self):
        child1 = Node(len(self.nodes), self.current_node, self.current_node)
        child2 = Node(len(self.nodes) + 1, self.current_node, self.current_node.last_branching)
        self.nodes.append(child1)
        self.nodes.append(child2)
        self.current_node.add_children(child1, child2)
        self.current_node = child1

    def __add_data_to_current_node(self, column_name, column_value):
        self.current_node.add_data(column_name, column_value)

    def __operations_until_now(self):
        parent_node = self.current_node.parent
        current_node = self.current_node
        columns = []
        values = []
        which_children = []
        while parent_node != None:
            columns.append(parent_node.column_name)
            values.append(parent_node.column_value)
            which_children.append(1 if parent_node.child1 == current_node else 2)
            parent_node, current_node = parent_node.parent, parent_node
        columns.reverse()
        values.reverse()
        which_children.reverse()
        return columns, values, which_children

    def __create_new_data_frame(self):
        data_frame_copy = self.data_frame.copy()
        columns, values, which_children = self.__operations_until_now()
        for (column, value, which_child) in zip(columns, values, which_children):
            data_frame_copy = data_frame_copy[data_frame_copy[column] < value] if which_child == 1 else data_frame_copy[data_frame_copy[column] >= value]
        return data_frame_copy

    def __go_to_next_leaf_if_needed(self):
        data_frame = self.__create_new_data_frame()
        while len(data_frame.index) <= 5:
            self.current_node.add_average(data_frame[self.target_col].mean())
            if self.current_node.last_branching != None:
                self.current_node = self.current_node.last_branching.child2
                data_frame = self.__create_new_data_frame()
            else:
                self.done_building_tree = True
                return

    def __loss_function(self, data_frame, column_name, column_value):
        mean_1 = data_frame[data_frame[column_name] <  column_value][self.target_col].mean()
        mean_2 = data_frame[data_frame[column_name] >= column_value][self.target_col].mean()
        summand_1 = data_frame[data_frame[column_name] <  column_value][self.target_col].apply(lambda x: (x - mean_1)**2).sum()
        summand_2 = data_frame[data_frame[column_name] >= column_value][self.target_col].apply(lambda x: (x - mean_2)**2).sum()
        return summand_1 + summand_2

    def __where_to_split(self, columns):
        data_frame = self.__create_new_data_frame()
        return min(((i, j) for (i, j) in it.product(columns, data_frame.index)), key = (lambda x: self.__loss_function(data_frame, x[0], data_frame.loc[x[1], x[0]])))

    def __build_tree(self):
        columns = self.data_frame.columns.tolist()
        columns.remove(self.target_col)
        while not self.done_building_tree:
            (col_to_split, val_to_split) = self.__where_to_split(columns)
            self.__add_data_to_current_node(col_to_split, val_to_split)
            self.__add_nodes()
            self.__go_to_next_leaf_if_needed()

    def __predict_record(self, record):
        data_frame = self.data_frame
        current_node = self.starting_node
        while not current_node.leaf:
            if record[current_node.column_name] < current_node.column_value:
                current_node = current_node.child1
            else:
                current_node = current_node.child2
        return current_node.average

    def predict(self, data_frame):
        return data_frame.apply(self.__predict_record, axis = 1)

train_df = pd.DataFrame({"a": [i for i in range(10)], "b": [int(i / 4) for i in range(10)], "c": [i % 4 for i in range(10)]})
DecTree = DecisionTree(train_df, "c")

test_df = pd.DataFrame({"a": [i for i in range(10, 20)], "b": [int(i / 4) for i in range(10)]})
test_df["Pred"] = DecTree.predict(test_df)
print(test_df)
