# Define "SpatialOmicsImageDataset" class based on ordinary Python list.
class SpatialOmicsWangxiao(InMemoryDataset):                                         
    def __init__(self, root, transform=None, pre_transform=None):
        super(SpatialOmicsWangxiao, self).__init__(root, transform, pre_transform)  
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['SpatialOmicsWangxiao.pt']                                           

    def download(self):
        pass
    
    def process(self):
        # Read data_list into huge `Data` list.
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# Create an object of this "SpatialOmicsImageDataset" class.
dataset = SpatialOmicsWangxiao(ThisStep_OutputFolderName) #transform=T.ToDense(max_nodes)
print("Step1 done!")
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))