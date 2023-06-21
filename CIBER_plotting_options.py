# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("jms_style_sheet")

# %%
ciber_df = pd.read_csv("figure_data/CIBER-plot-data-fullSample.csv").assign(
    **{"short_fullLabel": lambda d: d["fullLabel"].str.split("[").str[0]}
)
# %%
ciber_df.head()
# %%
df = pd.read_csv("figure_data/ciber_data_standardized.csv", index_col=0)
# %%
det_columns = ciber_df["label"].to_list()
# %%
df[det_columns].head()
# %%
# Option 1: Violin plot of CIBER scores
plot_df = df[det_columns].melt()
fig, ax_arr = plt.subplots(figsize=(10, 20), ncols=2)
ax = sns.boxenplot(
    x="value",
    y="variable",
    data=plot_df,
    color="white",
    order=ciber_df.sort_values(by="r_point", ascending=False)["label"].tolist(),
    ax=ax_arr[0],
)
label_name = "fullLabel"
plot_df = ciber_df[[label_name, "mean_ci_lo", "mean_point", "mean_ci_hi"]].melt(
    id_vars=label_name
)
mean_ax = sns.violinplot(
    data=plot_df,
    x="value",
    y=label_name,
    inner="stick",
    cut=0,
    order=ciber_df.sort_values(by="r_point", ascending=False)[label_name].tolist(),
    ax=ax_arr[0],
)
_ = ax.set_xlabel("Scores and 99.99% Confidence Intervals")
_ = ax.set_ylabel("")
plot_df = ciber_df[[label_name, "r_ci_lo", "r_point", "r_ci_hi"]].melt(
    id_vars=label_name
)
r_ax = sns.violinplot(
    data=plot_df,
    x="value",
    y=label_name,
    inner="stick",
    cut=0,
    order=ciber_df.sort_values(by="r_point", ascending=False)[label_name].tolist(),
    color="grey",
    ax=ax_arr[1],
)
_ = r_ax.set_xlabel("95% Confidence Intervals of Associations")
_ = r_ax.set_yticks([])
_ = r_ax.set_ylabel("")

# %%
# fig.savefig("figure/CIBER_plotting_options.png", dpi=300, bbox_inches="tight")
_ = fig.savefig('figure/CIBER_plotting_options.eps', format='eps', dpi=1000, bbox_inches='tight')
# %%
# Option 2: Density plot of CIBER scores
# plot_df = df[det_columns].melt()
# fig, ax_arr = plt.subplots(figsize=(10, 30), ncols=2)
# ax = sns.barplot(x="value",
#                 y="variable",
#                 data=plot_df,
#                 # color="white",
#                 order=ciber_df.sort_values(by="r_point", ascending=False)["label"].tolist(),
#                 ax=ax_arr[0]
#                 )
# label_name = "fullLabel"
# plot_df = ciber_df[[label_name, "mean_ci_lo", "mean_point", "mean_ci_hi"]].melt(id_vars=label_name)
# mean_ax = sns.violinplot(
#     data=plot_df,
#     x="value",
#     y=label_name,
#     inner="stick",
#     cut=0,
#     order=ciber_df.sort_values(by="r_point", ascending=False)[label_name].tolist(),
#     ax=ax_arr[0],
# )
# _ = ax.set_xlabel("Scores and 99.99% Confidence Intervals")
# _ = ax.set_ylabel("")

# plot_df = ciber_df[["fullLabel", "r_ci_lo", "r_point", "r_ci_hi"]].melt(id_vars=label_name)
# r_ax = sns.violinplot(
#     data=plot_df,
#     x="value",
#     y=label_name,
#     inner="stick",
#     cut=0,
#     order=ciber_df.sort_values(by="r_point", ascending=False)[label_name].tolist(),
#     color="grey",
#     ax=ax_arr[1],)
# _ = r_ax.set_xlabel("95% Confidence Intervals of Associations")
# _ = r_ax.set_yticks([])
# _ = r_ax.set_ylabel("")
