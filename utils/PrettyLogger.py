import pprint


def pretty_dict(d, indent=0, width=86):
    pp = pprint.PrettyPrinter(indent=indent, width=width, compact=False)
    return '\n'.join('\t\t' + line for line in pp.pformat(d).split('\n')).strip()


class PrettyLogger:
    def __init__(self, log_file_path=None):
        self.log_file_path = log_file_path
        f = None
        if log_file_path is not None:
            try:
                f = open(self.log_file_path, mode='a', encoding='utf-8')
            except:
                f = open(self.log_file_path, mode='w', encoding='utf-8')
            finally:
                if f is not None:
                    f.write("#" * 100)
                    f.close()
                else:
                    self.log_file_path = None

    def print_out(self, output):
        if self.log_file_path is None:
            print(output)
        else:
            try:
                with open(self.log_file_path, mode='a', encoding='utf-8') as f:
                    f.write("\n{output}\n".format(output=output))
                    f.close()
            except:
                print("Could't open file: " + self.log_file_path)

    def log_results(self, text_analysis, clf_type, metrics, best_params="Not stated",
                    feature_importances="Not stated"):
        self.print_out('''
      +---------------------------------------------------------------------------------+
      | type: {clf_type:<74}|
      | text_analysis: {text_analysis:<65}|
      | metrics:  +-----+------------------+------------------+------------------+      |
      |           |_____|________f1________|______roc_auc_____|_____business_____|      |
      |           |val: |{f1_u: <18}|{roc_auc_u: <18}|{business_u: <18}|      |
      |           +-----+------------------+------------------+------------------+      |
      | time: {time_u:<74}|
      | best_params: {best_params}
      | feature_importances: {feature_importances}
      |                                                                                 |
      +---------------------------------------------------------------------------------+
      '''.format(clf_type=clf_type,
                 text_analysis=str(text_analysis),
                 f1_u=round(metrics[0], 3), roc_auc_u=round(metrics[1], 3), business_u=round(metrics[2], 3),
                 time_u=round(metrics[3], 3),
                 best_params=pretty_dict(best_params),
                 feature_importances=pretty_dict(feature_importances)))
