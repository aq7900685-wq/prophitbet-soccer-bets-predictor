import pandas as pd
from datetime import date
from PyQt6.QtWidgets import QDialog, QLabel, QLineEdit, QComboBox, QHBoxLayout, QMessageBox, QPushButton, QVBoxLayout
from src.database.model import ModelDatabase
from src.gui.widgets.tables import SimpleTableDialog
from src.metrics.value import evaluate_value_bet, format_decision
from src.preprocessing.utils.target import TargetType
from src.preprocessing.utils.inputs import construct_inputs_by_teams


class PredictorDialog(QDialog):
    """ Predictor dialog which predicts the outcome of a match. """

    def __init__(self, df: pd.DataFrame, model_db: ModelDatabase):
        super().__init__()

        self._df = df.reset_index(drop=True)

        self._model_db = model_db
        self._model_ids = model_db.get_model_ids()
        self._title = 'Manual Prediction Dialog'
        self._width = 540
        self._height = 240

        self._home_teams = sorted(df['Home'].unique().tolist())
        self._away_teams = sorted(df['Away'].unique().tolist())
        self._result_dict = {0: 'H', 1: 'D', 2: 'A'}
        self._result_uo_dict = {0: 'U', 1: 'O'}

        # Declare placeholders.
        self._target_types = {'Result (1/X/2)': TargetType.RESULT, 'U/O-2.5': TargetType.OVER_UNDER}
        self._result_model_ids = []
        self._uo_model_ids = []
        for model_id in self._model_ids:
            config = model_db.load_model_config(model_id=model_id)
            if config['target_type'] == TargetType.RESULT:
                self._result_model_ids.append(model_id)
            else:
                self._uo_model_ids.append(model_id)

        # Declare UI Placeholders.
        self._combo_model = None
        self._combo_target = None
        self._combo_home = None
        self._combo_away = None
        self._edit_home_odd = None
        self._edit_draw_odd = None
        self._edit_away_odd = None
        self._edit_under_odd = None
        self._edit_over_odd = None
        self._edit_min_edge = None
        self._predict_btn = None

        self._initialize_window()
        self._add_widgets()

    def _initialize_window(self):
        self.setWindowTitle(self._title)
        self.resize(self._width, self._height)

    def _add_widgets(self):
        root = QVBoxLayout(self)
        root.addSpacing(15)

        # --- Model initialization ---
        model_hbox = QHBoxLayout()
        model_hbox.addStretch(1)
        self._combo_target = QComboBox()
        self._combo_target.setFixedWidth(120)
        for target in self._target_types:
            self._combo_target.addItem(target)
        self._combo_target.setCurrentIndex(-1)
        self._combo_target.currentIndexChanged.connect(self._on_target_change)
        model_hbox.addWidget(QLabel(' Target: '))
        model_hbox.addWidget(self._combo_target)

        self._combo_model = QComboBox()
        self._combo_model.setFixedWidth(220)
        self._combo_model.setCurrentIndex(-1)
        self._combo_model.currentIndexChanged.connect(self._on_model_change)
        model_hbox.addWidget(QLabel(' Model ID: '))
        model_hbox.addWidget(self._combo_model)
        model_hbox.addStretch(1)
        root.addLayout(model_hbox)

        teams_hbox = QHBoxLayout()
        teams_hbox.addStretch(1)
        self._combo_home = QComboBox()
        self._combo_home.setFixedWidth(150)
        for team in self._home_teams:
            self._combo_home.addItem(team)
        self._combo_home.setCurrentIndex(-1)
        teams_hbox.addWidget(QLabel(text='Home Team'))
        teams_hbox.addWidget(self._combo_home)
        teams_hbox.addWidget(QLabel(' vs '))

        self._combo_away = QComboBox()
        self._combo_away.setFixedWidth(150)
        for team in self._away_teams:
            self._combo_away.addItem(team)
        self._combo_away.setCurrentIndex(-1)
        teams_hbox.addWidget(QLabel(text='Away Team'))
        teams_hbox.addWidget(self._combo_away)
        teams_hbox.addStretch(1)
        root.addLayout(teams_hbox)

        odds_hbox = QHBoxLayout()
        odds_hbox.addStretch(1)
        self._edit_home_odd = QLineEdit(text='1.00')
        self._edit_home_odd.setFixedWidth(60)
        self._edit_home_odd.setPlaceholderText('Home Odd...')
        odds_hbox.addWidget(QLabel('1:'))
        odds_hbox.addWidget(self._edit_home_odd)

        self._edit_draw_odd = QLineEdit(text='1.00')
        self._edit_draw_odd.setFixedWidth(60)
        self._edit_draw_odd.setPlaceholderText('Home Odd...')
        odds_hbox.addWidget(QLabel('X:'))
        odds_hbox.addWidget(self._edit_draw_odd)

        self._edit_away_odd = QLineEdit(text='1.00')
        self._edit_away_odd.setFixedWidth(60)
        self._edit_away_odd.setPlaceholderText('Away Odd...')
        odds_hbox.addWidget(QLabel('2:'))
        odds_hbox.addWidget(self._edit_away_odd)
        odds_hbox.addStretch(1)
        root.addLayout(odds_hbox)

        # Over/Under odds (used by the value/decision ruling for the U/O-2.5 target only) + min edge.
        decision_hbox = QHBoxLayout()
        decision_hbox.addStretch(1)
        self._edit_under_odd = QLineEdit(text='1.00')
        self._edit_under_odd.setFixedWidth(60)
        self._edit_under_odd.setPlaceholderText('Under Odd...')
        decision_hbox.addWidget(QLabel('U:'))
        decision_hbox.addWidget(self._edit_under_odd)

        self._edit_over_odd = QLineEdit(text='1.00')
        self._edit_over_odd.setFixedWidth(60)
        self._edit_over_odd.setPlaceholderText('Over Odd...')
        decision_hbox.addWidget(QLabel('O:'))
        decision_hbox.addWidget(self._edit_over_odd)

        self._edit_min_edge = QLineEdit(text='5')
        self._edit_min_edge.setFixedWidth(50)
        self._edit_min_edge.setToolTip('Minimum expected-value edge (%) required to rule BET instead of SHOP/SKIP.')
        decision_hbox.addWidget(QLabel('  Min Edge %:'))
        decision_hbox.addWidget(self._edit_min_edge)
        decision_hbox.addStretch(1)
        root.addLayout(decision_hbox)

        self._predict_btn = QPushButton('Predict')
        self._predict_btn.setFixedWidth(100)
        self._predict_btn.setFixedHeight(30)
        self._predict_btn.clicked.connect(self._predict)
        self._predict_btn.setEnabled(False)
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 10, 0, 0)
        btn_row.addStretch(1)
        btn_row.addWidget(self._predict_btn)
        btn_row.addStretch(1)
        root.addLayout(btn_row)
        root.addStretch(1)

    def exec(self):
        if len(self._model_ids) == 0:
            QMessageBox.critical(
                self,
                'No Existing Models.',
                'There are no existing models to predict.',
                QMessageBox.StandardButton.Ok
            )
            return QDialog.Rejected

        super().exec()

    def _on_target_change(self):
        """ Adds model ids based on the selected target. """

        # Disable Predict button.
        self._predict_btn.setEnabled(False)

        # Clearing model ids.
        self._combo_model.blockSignals(True)
        self._combo_model.clear()

        # Setting model. percentiles.
        target_type = self._target_types[self._combo_target.currentText()]

        if target_type == TargetType.RESULT:
            model_ids = self._result_model_ids
        elif target_type == TargetType.OVER_UNDER:
            model_ids = self._uo_model_ids
        else:
            raise ValueError(f'Undefined targets: "{target_type}"')

        # Adding model ids.
        for model_id in model_ids:
            self._combo_model.addItem(model_id)
        self._combo_model.setCurrentIndex(-1)
        self._combo_model.blockSignals(False)

    def _on_model_change(self):
        self._predict_btn.setEnabled(True)

    def _predict(self):
        if not self._validate_inputs():
            return

        # Constructing model input.
        match_df = pd.DataFrame({
            'Date': [date.today().strftime(format='%Y-%m-%d')],
            'Home': [self._combo_home.currentText()],
            'Away': [self._combo_away.currentText()],
            '1': [float(self._edit_home_odd.text().strip())],
            'X': [float(self._edit_draw_odd.text().strip())],
            '2': [float(self._edit_away_odd.text().strip())]
        })
        match_df = construct_inputs_by_teams(df=self._df, match_df=match_df)

        # Loading model.
        model_id = self._combo_model.currentText()
        model, _ = self._model_db.load_model(model_id=model_id)

        # Generating Predictions.
        y_prob = model.predict_proba(df=match_df)
        y_pred = y_prob.argmax(axis=1)
        y_prob = y_prob[0].round(2)

        # Show table dialog.
        match_df = match_df[['Date', 'Season', 'Week', 'Home', 'Away', '1', 'X', '2']]

        # Adding predictions to Dataframe.
        target_type = self._target_types[self._combo_target.currentText()]
        min_edge = float(self._edit_min_edge.text().strip())/100.0

        if target_type == TargetType.RESULT:
            match_df['Predicted'] = self._result_dict[y_pred[0]]
            match_df['Prob(1)'] = y_prob[0]
            match_df['Prob(X)'] = y_prob[1]
            match_df['Prob(2)'] = y_prob[2]

            # Value/decision ruling: decide on expected value vs the offered 1/X/2 odds, not on win%.
            decision = evaluate_value_bet(
                probs=y_prob,
                odds=match_df[['1', 'X', '2']].to_numpy(dtype=float)[0],
                labels=['1', 'X', '2'],
                min_edge=min_edge
            )
        elif target_type == TargetType.OVER_UNDER:
            match_df['Predicted'] = self._result_uo_dict[y_pred[0]]
            match_df['Prob(U)'] = y_prob[0]
            match_df['Prob(O)'] = y_prob[1]

            # Value/decision ruling using the manually-entered Under/Over odds.
            decision = evaluate_value_bet(
                probs=y_prob,
                odds=[float(self._edit_under_odd.text().strip()), float(self._edit_over_odd.text().strip())],
                labels=['U', 'O'],
                min_edge=min_edge
            )
        else:
            raise ValueError(f'Undefined target type: "{target_type}".')

        match_df['Value'] = f'{decision["pick"]}{"" if decision["consistent"] else "*"}'
        match_df['EV%'] = round(decision['ev']*100.0, 1)
        match_df['Edge%'] = round(decision['edge']*100.0, 1)
        match_df['Decision'] = format_decision(decision)

        SimpleTableDialog(df=match_df, parent=self, title='Prediction', readonly=False).show()

    def _validate_inputs(self) -> bool:
        # 1/X/2 odds are always required (they are model input features).
        odds = [
            self._edit_home_odd.text(),
            self._edit_draw_odd.text(),
            self._edit_away_odd.text(),
        ]

        # Under/Over odds are required only for the U/O-2.5 target (used by the decision ruling).
        target_type = self._target_types.get(self._combo_target.currentText())
        if target_type == TargetType.OVER_UNDER:
            odds += [self._edit_under_odd.text(), self._edit_over_odd.text()]

        for odd in odds:
            try:
                odd = float(odd.strip())
            except ValueError:
                QMessageBox.critical(self, 'Invalid Odds', f'Odd "{odd}" is not numeric.')
                return False
            else:
                if odd <= 1.0:
                    QMessageBox.critical(self, 'Invalid Odds', f'Odds cannot be less than 1.00, found "{odd}".')
                    return False

        # Min edge must be a non-negative percentage.
        try:
            min_edge = float(self._edit_min_edge.text().strip())
        except ValueError:
            QMessageBox.critical(self, 'Invalid Min Edge', f'Min Edge "{self._edit_min_edge.text()}" is not numeric.')
            return False
        else:
            if min_edge < 0.0:
                QMessageBox.critical(self, 'Invalid Min Edge', f'Min Edge cannot be negative, found "{min_edge}".')
                return False
        return True
