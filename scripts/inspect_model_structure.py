from cmid_qa.custom_encdec import CustomMultiTaskTransformer


def inspect_structure():
    model = CustomMultiTaskTransformer(
        vocab_size=1000,
        num_classes_4=4,
        num_classes_36=36,
        d_model=256,
        nhead=4,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=512,
    )
    print(model)


if __name__ == "__main__":
    inspect_structure()
