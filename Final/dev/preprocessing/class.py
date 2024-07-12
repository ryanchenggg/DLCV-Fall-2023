import json

def check_classes(json_file):
    with open(json_file) as f:
        data = json.load(f)
    classes = []
    for class_info in data['categories']:
        classes.append(class_info['name'])
    classes = list(set(classes))

    return classes

def fix_class_id(json_file):
    with open(json_file) as f:
        data = json.load(f)
    # Deak with id_sets
    classes = []
    id_sets = dict()
    for class_info in data['categories']:
        if class_info['name'] in classes:
            id_sets[class_info['name']].add(class_info['id'])
        else:
            classes.append(class_info['name'])
            id_sets[class_info['name']] = set([class_info['id']])

    print(len(id_sets.keys()))
    # Save the new "categories"
    new_categories = []
    for i, name in enumerate(id_sets.keys()):
        new_categories.append({
            "id": i,
            "name": name
        })
    
    data['categories'] = new_categories
    print(data['categories'])

    # Deal with json['annotations']
    for annotation in data['annotations']:
        for i, name in enumerate(id_sets.keys()):
            if annotation['category_id'] in id_sets[name]:
                annotation['category_id'] = i
                break   
    
    # Save the new json file
    with open(json_file.split('.')[0] + '_fixed.json', 'w') as f:
        json.dump(data, f, indent=4)
    

if __name__ == "__main__":
    
    # For this json file, keys are 'categories', 'images', 'annotations' 
    f = [
        "/home/ryan/dlcv/DLCV-Fall-2023-Final-2-boss-sdog/DLCV_vq2d_data/objects_train.json", 
        "/home/ryan/dlcv/DLCV-Fall-2023-Final-2-boss-sdog/DLCV_vq2d_data/objects_val.json",
    ]
    
    fix_class_id(f[1])
    # print(len(classes_unique), '\n', type(classes_unique))

    # print(len(classes_train))
    # print(len(classes), '\n', classes)