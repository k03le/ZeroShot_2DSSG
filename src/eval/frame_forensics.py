"""
Script:  frame_forensics.py
Purpose: Per-frame forensic dump answering "at which stage did this fail?"
         Unlike the earlier pull, this shows FAILED matches too, with the best
         IoU each predicted box achieved -- so a near-miss (IoU 0.45) is
         distinguishable from a total miss (IoU 0.02), and absorption --
         a box matching the wrong GT instance confidently -- is visible as such.

         For each requested frame it prints:
           1. predicted objects, and for EACH: best GT candidate + IoU + verdict
           2. GT instances, and for EACH: best predicted candidate + IoU + verdict
           3. every VLM relation, with both endpoints' match status
           4. every canonical GT edge, with why it was/wasn't scoreable

READ-ONLY.
USAGE
    python frame_forensics.py \
        --frames-json /data/mmc-user/outputs/3dssg_upright_v2/ingest_identify.json \
        --dssg-root /data/mmc-user/3dssg \
        --rel-dir /data/mmc-user/outputs/relations_3dssg_vlmfirst/json \
        --frames 000199,000175,000171,000117,000110,000030
"""
import argparse, glob, json, os
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

PREDICATE_VOCAB = {
    "left","right","close by","attached to","standing on","higher than",
    "lower than","same as","lying on","hanging on","supported by",
    "standing in","build in","leaning against","connected to",
    "belonging to","lying in","part of","cover","hanging in"}
SYMMETRIC = {"close by","same as"}
INVERSE = {"left":"right","right":"left",
           "higher than":"lower than","lower than":"higher than"}

def canonical_gt(s,o,p):
    if p not in PREDICATE_VOCAB: return None
    if p in SYMMETRIC: return (min(s,o),max(s,o),p)
    if p in INVERSE: return (s,o,p) if s<=o else (o,s,INVERSE[p])
    return (s,o,p)

def iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0.,ix2-ix1),max(0.,iy2-iy1); inter=iw*ih
    aa=max(0.,ax2-ax1)*max(0.,ay2-ay1); bb=max(0.,bx2-bx1)*max(0.,by2-by1)
    u=aa+bb-inter
    return inter/u if u>0 else 0.

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--frames-json",required=True)
    ap.add_argument("--dssg-root",default="/data/mmc-user/3dssg")
    ap.add_argument("--data-subdir",default="3dssg_data")
    ap.add_argument("--rel-dir",required=True)
    ap.add_argument("--frames",required=True,help="comma-separated frame numbers, e.g. 000199,000175")
    ap.add_argument("--min-px",type=int,default=200)
    ap.add_argument("--iou-thresh",type=float,default=0.5)
    args=ap.parse_args()
    want={x.strip() for x in args.frames.split(",")}

    root=Path(args.dssg_root)
    rels_raw=json.loads((root/"relationships.json").read_text())
    rels_by_scan={s["scan"]:s["relationships"] for s in rels_raw["scans"]}
    objs_raw=json.loads((root/"objects.json").read_text())
    labels_by_scan={s["scan"]:{int(o["id"]):o.get("label","?") for o in s["objects"]}
                    for s in objs_raw["scans"]}
    rep=json.loads(Path(args.frames_json).read_text())
    frames=[f for f in rep["frames"] if f.get("ok") and "{0:06d}".format(f["frame_no"]) in want]

    for f in sorted(frames,key=lambda r:(r["scan"],r["frame_no"])):
        scan,no=f["scan"],f["frame_no"]
        lab=labels_by_scan.get(scan,{})
        print("\n"+"="*78)
        print("FRAME {0}_{1:06d}".format(scan,no))
        print("="*78)

        mpath=root/args.data_subdir/scan/"sequence"/"frame-{0:06d}.rendered.instances.png".format(no)
        mask=np.asarray(Image.open(mpath))
        ids,counts=np.unique(mask,return_counts=True)
        gt_boxes={}
        for oid,cnt in zip(ids,counts):
            if oid==0 or cnt<args.min_px: continue
            ys,xs=np.where(mask==oid)
            gt_boxes[int(oid)]=(float(xs.min()),float(ys.min()),float(xs.max()),float(ys.max()))

        m=sorted(glob.glob(os.path.join(args.rel_dir,"{0}_{1:06d}__*.json".format(scan,no))))
        if not m:
            print("  no relations file"); continue
        rd=json.loads(Path(m[0]).read_text())
        pred_boxes=[tuple(o["bbox_2d"]) for o in rd["objects"]]
        tags=[o["tag"] for o in rd["objects"]]
        gt_ids=list(gt_boxes.keys())

        im=np.zeros((len(pred_boxes),len(gt_ids)))
        for pi,pb in enumerate(pred_boxes):
            for gi,gid in enumerate(gt_ids):
                im[pi,gi]=iou(pb,gt_boxes[gid])
        box_to_gt={}
        if len(pred_boxes) and len(gt_ids):
            ri,ci=linear_sum_assignment(1.0-im)
            for pi,gi in zip(ri,ci):
                if im[pi,gi]>=args.iou_thresh:
                    box_to_gt[pi]=gt_ids[gi]

        print("\n-- 1. PREDICTED boxes: best GT candidate, IoU, verdict --")
        for pi,t in enumerate(tags):
            if len(gt_ids)==0: continue
            gi=int(np.argmax(im[pi])); best=im[pi,gi]
            v="MATCHED to {0}:{1}".format(box_to_gt[pi],lab.get(box_to_gt[pi],"?")) if pi in box_to_gt \
              else "UNMATCHED (best {0}:{1} IoU={2:.2f})".format(gt_ids[gi],lab.get(gt_ids[gi],"?"),best)
            print("  {0:>3}: {1:<34} {2}".format(pi,t[:34],v))

        print("\n-- 2. GT instances: best predicted candidate, IoU, verdict --")
        matched_gt=set(box_to_gt.values())
        for gi,gid in enumerate(gt_ids):
            pi=int(np.argmax(im[:,gi])) if len(pred_boxes) else None
            best=im[pi,gi] if pi is not None else 0.0
            v="matched" if gid in matched_gt else "UNMATCHED (best pred '{0}' IoU={1:.2f})".format(
                tags[pi][:26] if pi is not None else "-",best)
            print("  {0:>4}:{1:<20} {2}".format(gid,lab.get(gid,"?")[:20],v))

        print("\n-- 3. VLM relations, endpoint match status --")
        for r in rd.get("relations_readable",[]):
            si=int(str(r["subject"]).split(":",1)[0]); oi=int(str(r["object"]).split(":",1)[0])
            ss="OK->{0}".format(box_to_gt[si]) if si in box_to_gt else "unmatched"
            os_="OK->{0}".format(box_to_gt[oi]) if oi in box_to_gt else "unmatched"
            usable="BOTH OK" if (si in box_to_gt and oi in box_to_gt) else "-- dead"
            print("  {0:<28} --{1:<22}--> {2:<28}  [{3} | {4}]  {5}".format(
                str(r["subject"])[:28],str(r["relation"])[:22],str(r["object"])[:28],ss,os_,usable))

        print("\n-- 4. canonical GT edges: scoreable? --")
        by_pair=defaultdict(list)
        for r in rd.get("relations_readable",[]):
            si=int(str(r["subject"]).split(":",1)[0]); oi=int(str(r["object"]).split(":",1)[0])
            if si in box_to_gt and oi in box_to_gt:
                a,b=box_to_gt[si],box_to_gt[oi]
                if a!=b: by_pair[frozenset((a,b))].append((a,b,r["relation"]))
        visible=set(gt_ids); seen=set()
        for e in rels_by_scan.get(scan,[]):
            s_id,o_id,_pid,pname=int(e[0]),int(e[1]),e[2],str(e[3])
            if s_id not in visible or o_id not in visible: continue
            can=canonical_gt(s_id,o_id,pname)
            if can is None or can in seen: continue
            seen.add(can); cs,co,cp=can
            cands=by_pair.get(frozenset((cs,co)),[])
            if not cands:
                why="no VLM relation on this grounded pair"
            else:
                sg,og,phr=cands[0]
                if (sg,og)==(cs,co): why="SCOREABLE via {0!r}".format(phr)
                elif cp in SYMMETRIC or cp in INVERSE: why="SCOREABLE (orientation-equivalent) via {0!r}".format(phr)
                else: why="DIRECTION ERROR via {0!r}".format(phr)
            print("  {0}:{1} --{2}--> {3}:{4}".format(cs,lab.get(cs,"?"),cp,co,lab.get(co,"?")))
            print("       {0}".format(why))

if __name__=="__main__":
    main()