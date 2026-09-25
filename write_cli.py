import os

with open('researchbench/cli.py', 'r', encoding='utf-8') as f:
    cli = f.read()

# Replace hardcoded evaluation
old_eval = '''        elif args.command == "evaluate":
            y = df[args.target]
            import numpy as np
            X = df.drop(columns=[args.target]).select_dtypes(include=[np.number])
            # Impute missing values for basic evaluation safety
            X = X.fillna(X.mean())
            res = evaluate_models(X, y, args.task, args.models)'''

new_eval = '''        elif args.command == "evaluate":
            y = df[args.target]
            X = df.drop(columns=[args.target])
            from researchbench.config import load_config
            config = load_config(getattr(args, 'config', None))
            preprocess_mode = getattr(args, 'preprocess', 'auto')
            res = evaluate_models(X, y, args.task, args.models, config=config, preprocess_mode=preprocess_mode)'''

cli = cli.replace(old_eval, new_eval)

# Replace hardcoded CV
old_cv = '''        elif args.command == "cv":
            y = df[args.target]
            import numpy as np
            X = df.drop(columns=[args.target]).select_dtypes(include=[np.number])
            X = X.fillna(X.mean())
            from researchbench.evaluation.classification import get_classification_models'''

new_cv = '''        elif args.command == "cv":
            y = df[args.target]
            X = df.drop(columns=[args.target])
            from researchbench.config import load_config
            config = load_config(getattr(args, 'config', None))
            preprocess_mode = getattr(args, 'preprocess', 'auto')
            
            from researchbench.evaluation.preprocessing import build_model_pipeline
            from researchbench.evaluation.classification import get_classification_models'''

cli = cli.replace(old_cv, new_cv)

# Replace hardcoded Stability
old_stability = '''        elif args.command == "stability":
            y = df[args.target]
            import numpy as np
            X = df.drop(columns=[args.target]).select_dtypes(include=[np.number])
            X = X.fillna(X.mean())
            from researchbench.evaluation.classification import get_classification_models'''

new_stability = '''        elif args.command == "stability":
            y = df[args.target]
            X = df.drop(columns=[args.target])
            from researchbench.config import load_config
            config = load_config(getattr(args, 'config', None))
            preprocess_mode = getattr(args, 'preprocess', 'auto')
            
            from researchbench.evaluation.preprocessing import build_model_pipeline
            from researchbench.evaluation.classification import get_classification_models'''

cli = cli.replace(old_stability, new_stability)

with open('researchbench/cli.py', 'w', encoding='utf-8') as f:
    f.write(cli)