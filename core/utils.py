import cv2

def draw_lanes(frame, lanes):
    for lane in lanes:
        # line1
        cv2.line(
            frame,
            tuple(lane["line1"][0]),
            tuple(lane["line1"][1]),
            (0, 255, 255), 2
        )

        # line2
        cv2.line(
            frame,
            tuple(lane["line2"][0]),
            tuple(lane["line2"][1]),
            (255, 0, 255), 2
        )

        # lane id label
        cv2.putText(
            frame,
            lane["id"],
            tuple(lane["line1"][0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


def draw_tracks(frame, boxes, ids, track_states):
    for box, oid in zip(boxes, ids):
        x1, y1, x2, y2 = box

        color = (0, 255, 0)
        label = f"ID {oid}"

        st = track_states.get(oid)
        if st:
            if st.finalized:
                color = (0, 0, 255)
                label += " FINAL"
            elif st.crossed_l1_ts:
                color = (0, 165, 255)
                label += " L1"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
