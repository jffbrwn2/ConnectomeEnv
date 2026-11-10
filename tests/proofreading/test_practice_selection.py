from scripts.proofreading.prepare_cb2_practice import select_cases


def test_selection_excludes_practice_identity_spatial_overlap_and_duplicates():
    def row(key,kind,root,point,split='test',op=None):
        return {'combined_sample_hash':key,'split':split,'sample_type':kind,
                'metadata':{'operation_id':op or key,'root_id':str(root),'interface_point_nm':point}}
    rows=[row('train','merge_edit',10,[0,0,0],'train'),
          row('a','merge_edit',10,[100,100,100]), # same root
          row('b','merge_edit',11,[1,1,1]), # overlaps
          row('c','merge_edit',12,[100,100,100]),
          row('d','split_edit',12,[200,200,200]), # identity reused within validation
          row('e','split_edit',13,[300,300,300])]
    audit={'operation_id':'train','sample_hash':'train','crop_bounds_nm':[[-5,-5,-5],[5,5,5]],
           'before_certified_composition':{'1':{'10':10}},'after_certified_composition':{}}
    result=select_cases(rows,{10,11,12,13},'test',1,10,[audit])
    assert [r['combined_sample_hash'] for r in result]==['c','e']
