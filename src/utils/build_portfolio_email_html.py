from pathlib import Path
from datetime import datetime
import pandas as pd


def build_roi_email_from_df(
    df: pd.DataFrame,
    action_list: list[str],
    image_dir: Path | str | None = None,
):
    """
    Build plain-text and HTML email bodies for ROI notifications
    from a portfolio DataFrame, including optional inline images.

    Expected DataFrame columns:
      ['Action', 'Purchase_Date', 'Purchase_price', 'Current_price',
       'Currency', 'Current_ROI', 'Target_ROI', 'ROI_reached']

    Image resolution:
      Action = "FRES"
      action_list contains "FRES.L_GBp→GBP"
      Image filename = "FRES.L_GBp→GBP.png"
    """

    now = datetime.today().strftime("%d-%m-%Y %H:%M")
    text_lines = [f"Data extracted on {now}\n"]
    table_rows = []
    image_sections = []
    inline_images = {}

    image_dir = Path(image_dir) if image_dir else None
    df = df.copy()

    # ---------- Build action lookup ----------
    action_lookup = {
        full.split(".")[0]: full for full in action_list
    }

    # ---------- Ensure datetime ----------
    df["Purchase_Date"] = pd.to_datetime(df["Purchase_Date"], errors="coerce")

    def row_color(current_roi: float, target_roi: float) -> str:
        if current_roi < 0:
            return "#f8d7da"   # red
        elif current_roi < target_roi:
            return "#fff3cd"   # yellow
        else:
            return "#d4edda"   # green

    for _, row in df.iterrows():
        action = row["Action"]
        cid = f"{action}_img"

        # ---------- TEXT BODY ----------
        text_lines.extend([
            f"Action: {action}",
            f"  Purchase Price: {row['Purchase_price']:.2f} {row['Currency']}",
            f"  Purchase Date: {row['Purchase_Date'].strftime('%d-%m-%Y')}",
            f"  Current Price: {row['Current_price']:.2f} {row['Currency']}",
            f"  Current ROI: {row['Current_ROI'] * 100:.2f}%",
            f"  Target ROI: {row['Target_ROI'] * 100:.2f}%",
            f"  ROI reached: {'YES' if row['ROI_reached'] else 'NO'}",
            "",
        ])

        # ---------- TABLE ROW ----------
        bg = row_color(row["Current_ROI"], row["Target_ROI"])

        table_rows.append(f"""
        <tr style="background-color:{bg};">
            <td><b>{action}</b></td>
            <td>{row['Purchase_Date'].strftime('%d-%m-%Y')}</td>
            <td>{row['Purchase_price']:.2f} {row['Currency']}</td>
            <td>{row['Current_price']:.2f} {row['Currency']}</td>
            <td>{row['Current_ROI'] * 100:.2f}%</td>
            <td>{row['Target_ROI'] * 100:.2f}%</td>
            <td>{'YES' if row['ROI_reached'] else 'NO'}</td>
        </tr>
        """)

        # ---------- IMAGE RESOLUTION ----------
        if image_dir:
            full_action = action_lookup.get(action)

            if full_action:
                img_path = image_dir / f"{full_action}_ROI.png"

                if img_path.exists():
                    inline_images[cid] = img_path
                    image_sections.append(f"""
                    <h4>{action} – Price evolution</h4>
                    <img src="cid:{cid}" style="max-width:900px; margin-bottom:20px;">
                    """)

    # ---------- HTML BODY ----------
    html_body = f"""
    <html>
    <body style="font-family: Arial, sans-serif;">
        <h2>Portfolio ROI status</h2>
        <p>Data extracted on <b>{now}</b></p>

        <table border="1" cellpadding="6" cellspacing="0"
               style="border-collapse:collapse;">
            <tr style="background:#f0f0f0; font-weight:bold;">
                <th>Action</th>
                <th>Purchase Date</th>
                <th>Purchase Price</th>
                <th>Current Price</th>
                <th>Current ROI</th>
                <th>Target ROI</th>
                <th>ROI reached</th>
            </tr>
            {''.join(table_rows)}
        </table>

        <hr>
        {''.join(image_sections)}
    </body>
    </html>
    """

    return "\n".join(text_lines), html_body, inline_images


if __name__ == "__main__":
    import os
    image_dir = os.path.join(os.getcwd(), "output")

    actions_list = ['III.L_GBp→GBP', 'ADM.L_GBp→GBP', 'AAF.L_GBp→GBP', 'ALW.L_GBp→GBP', 'AAL.L_GBp→GBP', 'ANTO.L_GBp→GBP', 'AHT.L_GBp→GBP', 'ABF.L_GBp→GBP', 'AZN.L_GBp→GBP', 'AUTO.L_GBp→GBP', 'AV.L_GBp→GBP', 'BAB.L_GBp→GBP', 'BA.L_GBp→GBP', 'BARC.L_GBp→GBP', 'BTRW.L_GBp→GBP', 'BEZ.L_GBp→GBP', 'BKG.L_GBp→GBP', 'BP.L_GBp→GBP', 'BATS.L_GBp→GBP', 'BLND.L_GBp→GBP', 'BT-A.L_GBp→GBP', 'BNZL.L_GBp→GBP', 'BRBY.L_GBp→GBP', 'CNA.L_GBp→GBP', 'CCEP.L_GBp→GBP', 'CCH.L_GBp→GBP', 'CPG.L_GBp→GBP', 'CTEC.L_GBp→GBP', 'CRDA.L_GBp→GBP', 'DCC.L_GBp→GBP', 'DGE.L_GBp→GBP', 'DPLM.L_GBp→GBP', 'EDV.L_GBp→GBP', 'ENT.L_GBp→GBP', 'EZJ.L_GBp→GBP', 'EXPN.L_GBp→GBP', 'FCIT.L_GBp→GBP', 'FRES.L_GBp→GBP', 'GAW.L_GBp→GBP', 'GLEN.L_GBp→GBP', 'GSK.L_GBp→GBP', 'HLN.L_GBp→GBP', 'HLMA.L_GBp→GBP', 'HIK.L_GBp→GBP', 'HSX.L_GBp→GBP', 'HWDN.L_GBp→GBP', 'HSBA.L_GBp→GBP', 'ICG.L_GBp→GBP', 'IHG.L_USD→GBP', 'IMI.L_GBp→GBP', 'IMB.L_GBp→GBP', 'INF.L_GBp→GBP', 'IAG.L_GBp→GBP', 'ITRK.L_GBp→GBP', 'JD.L_GBp→GBP', 'KGF.L_GBp→GBP', 'LAND.L_GBp→GBP', 'LGEN.L_GBp→GBP', 'LLOY.L_GBp→GBP', 'LMP.L_GBp→GBP', 'LSEG.L_GBp→GBP', 'MNG.L_GBp→GBP', 'MKS.L_GBp→GBP', 'MRO.L_GBp→GBP', 'MTLN.L_EUR→GBP', 'MNDI.L_GBp→GBP', 'NG.L_GBp→GBP', 'NWG.L_GBp→GBP', 'NXT.L_GBp→GBP', 'PSON.L_GBp→GBP', 'PSH.L_GBp→GBP', 'PSN.L_GBp→GBP', 'PHNX.L_GBp→GBP', 'PCT.L_GBp→GBP', 'PRU.L_GBp→GBP', 'RKT.L_GBp→GBP', 'REL.L_GBp→GBP', 'RTO.L_GBp→GBP', 'RMV.L_GBp→GBP', 'RIO.L_GBp→GBP', 'RR.L_GBp→GBP', 'SGE.L_GBp→GBP', 'SBRY.L_GBp→GBP', 'SDR.L_GBp→GBP', 'SMT.L_GBp→GBP', 'SGRO.L_GBp→GBP', 'SVT.L_GBp→GBP', 'SHEL.L_GBp→GBP', 'SMIN.L_GBp→GBP', 'SN.L_GBp→GBP', 'SPX.L_GBp→GBP', 'SSE.L_GBp→GBP', 'STAN.L_GBp→GBP', 'STJ.L_GBp→GBP', 'TSCO.L_GBp→GBP', 'ULVR.L_GBp→GBP', 'UU.L_GBp→GBP', 'VOD.L_GBp→GBP', 'WEIR.L_GBp→GBP', 'WTB.L_GBp→GBP']
    pur = pd.read_csv(os.path.join(os.getcwd(), "PORTFOLIO", "purchases.csv"))
    text_body, html_body, inline_images = build_roi_email_from_df(
    df = pur,
    action_list = actions_list,
    image_dir = image_dir,
)
    print(f"text_body: {text_body}\nhtml_body: {html_body}\ninline_images keys: {list(inline_images.keys())}\n")
    print(os.listdir(image_dir))